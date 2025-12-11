from flask import Flask, render_template, request, Response, send_file
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# 업로드된 영상 경로 / 설정값 전역 저장
CURRENT_VIDEO_PATH = None
CURRENT_MODEL_PATH = "yolo11n.pt"
CURRENT_CONF = 0.25
CURRENT_IOU = 0.45


# -----------------------------
# YOLO 모델 로드 (간단 캐시)
# -----------------------------
def load_model(path: str):
    """모델 경로가 바뀌면 다시 로드, 아니면 기존 모델 재사용"""
    global CURRENT_MODEL
    if "CURRENT_MODEL" not in globals() or getattr(CURRENT_MODEL, "model_path", None) != path:
        CURRENT_MODEL = YOLO(path)
        CURRENT_MODEL.model_path = path
    return CURRENT_MODEL


# -----------------------------
# 메인 화면
# -----------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        image_url=None,
        stream_url=None,
        model_path=CURRENT_MODEL_PATH,
        conf=CURRENT_CONF,
        iou=CURRENT_IOU,
    )


# -----------------------------
# 업로드 후 처리
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    global CURRENT_VIDEO_PATH, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU

    f = request.files.get("file")
    if not f:
        # 파일이 없으면 그냥 현재 설정값 유지한 채로 다시 렌더
        return render_template(
            "index.html",
            image_url=None,
            stream_url=None,
            model_path=CURRENT_MODEL_PATH,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
        )

    # 폼에서 설정값 읽어서 전역 업데이트
    CURRENT_MODEL_PATH = request.form.get("model_path", "yolo11n.pt")
    CURRENT_CONF = float(request.form.get("conf", 0.25))
    CURRENT_IOU = float(request.form.get("iou", 0.45))

    filename = f.filename.lower()

    # ---------- 이미지일 때 ----------
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # 이미지 열기
        img = Image.open(f.stream).convert("RGB")

        # 모델 로드
        model = load_model(CURRENT_MODEL_PATH)

        # 추론
        results = model.predict(
            source=np.array(img),
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
            verbose=False,
        )

        # 박스 그리기 (BGR)
        plotted = results[0].plot()
        # RGB로 변환
        plotted = plotted[:, :, ::-1]

        # 임시 파일로 저장
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.fromarray(plotted).save(out.name)

        # 결과 이미지 경로 템플릿으로 전달 + 설정값 유지
        return render_template(
            "index.html",
            image_url="/image_result?path=" + out.name,
            stream_url=None,
            model_path=CURRENT_MODEL_PATH,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
        )

    # ---------- 영상일 때 ----------
    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # 업로드 파일을 임시 mp4로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.save(tfile.name)
        CURRENT_VIDEO_PATH = tfile.name  # 전역에 기억 → 스트리밍에서 사용

        # 이미지 대신 스트리밍 URL 전달
        return render_template(
            "index.html",
            image_url=None,
            stream_url="/video_feed",
            model_path=CURRENT_MODEL_PATH,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
        )

    # 그 외 확장자면 그냥 다시 렌더
    return render_template(
        "index.html",
        image_url=None,
        stream_url=None,
        model_path=CURRENT_MODEL_PATH,
        conf=CURRENT_CONF,
        iou=CURRENT_IOU,
    )


# -----------------------------
# 이미지 결과 파일 서빙
# -----------------------------
@app.route("/image_result")
def image_result():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "No image", 404
    return send_file(path, mimetype="image/jpeg")


# -----------------------------
# 클래스 이름 재정의
# -----------------------------
NEW_CLASS_NAMES = {
    0: "승용차",
    1: "소형버스",
    2: "대형버스",
    3: "트럭",
    4: "대형트레일러",
    5: "오토바이",
    6: "보행자",
}

# -----------------------------
# 영상 프레임 스트리밍 제너레이터
# -----------------------------
def generate_video_stream():
    global CURRENT_VIDEO_PATH, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU

    if not CURRENT_VIDEO_PATH:
        return

    cap = cv2.VideoCapture(CURRENT_VIDEO_PATH)
    model = load_model(CURRENT_MODEL_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임마다 YOLO 추론
        results = model.predict(
            source=frame,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
            verbose=False,
        )
        
        if NEW_CLASS_NAMES:
                    results[0].names = NEW_CLASS_NAMES

        plotted = results[0].plot()  # BGR 프레임 (박스 포함)

        # JPEG로 인코딩해서 스트림 전송
        ok, buf = cv2.imencode(".jpg", plotted)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

    cap.release()


# -----------------------------
# 영상 스트림 엔드포인트
# -----------------------------
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# -----------------------------
# 엔트리 포인트
# -----------------------------
if __name__ == "__main__":
    # debug=False 로 두면 watchdog 에러 없이 깔끔하게 돌아감
    app.run(host="0.0.0.0", port=5000, debug=False)
