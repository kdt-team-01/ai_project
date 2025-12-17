from flask import Flask, render_template, request, Response, send_file
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------
# PyInstaller 대응
# ----------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, "templates"),
    static_folder=os.path.join(base_dir, "static")
)

# ----------------------------------------------------
# 전역 상태
# ----------------------------------------------------
CURRENT_VIDEO_PATH = None
CURRENT_MODEL_PATH = "best.pt"
CURRENT_CONF = 0.25
CURRENT_IOU = 0.45
CURRENT_MODEL = None

# ----------------------------------------------------
# 클래스 이름 재정의
# ----------------------------------------------------
NEW_CLASS_NAMES = {
    0: "승용차",
    1: "소형버스",
    2: "대형버스",
    3: "트럭",
    4: "대형트레일러",
    5: "오토바이",
    6: "보행자",
}

# ----------------------------------------------------
# 모델 로드 (캐시)
# ----------------------------------------------------
def load_model(path: str):
    global CURRENT_MODEL

    if CURRENT_MODEL is None or getattr(CURRENT_MODEL, "model_path", None) != path:
        CURRENT_MODEL = YOLO(path)
        CURRENT_MODEL.model_path = path

        if NEW_CLASS_NAMES:
            CURRENT_MODEL.names = NEW_CLASS_NAMES

    return CURRENT_MODEL

# ----------------------------------------------------
# 메인 페이지
# ----------------------------------------------------
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

# ----------------------------------------------------
# 업로드 처리
# ----------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    global CURRENT_VIDEO_PATH, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU

    f = request.files.get("file")
    if not f:
        return index()

    CURRENT_MODEL_PATH = request.form.get("model_path", CURRENT_MODEL_PATH)
    CURRENT_CONF = float(request.form.get("conf", CURRENT_CONF))
    CURRENT_IOU = float(request.form.get("iou", CURRENT_IOU))

    filename = f.filename.lower()
    model = load_model(CURRENT_MODEL_PATH)

    # ---------- 이미지 ----------
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(f.stream).convert("RGB")

        results = model.predict(
            source=np.array(img),
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
            verbose=False,
        )

        if NEW_CLASS_NAMES:
            results[0].names = NEW_CLASS_NAMES
            if hasattr(model, "predictor"):
                model.predictor.model.names = NEW_CLASS_NAMES

        plotted = results[0].plot()
        plotted = plotted[:, :, ::-1]

        out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.fromarray(plotted).save(out.name)

        return render_template(
            "index.html",
            image_url="/image_result?path=" + out.name,
            stream_url=None,
            model_path=CURRENT_MODEL_PATH,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
        )

    # ---------- 영상 ----------
    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.save(tfile.name)
        CURRENT_VIDEO_PATH = tfile.name

        return render_template(
            "index.html",
            image_url=None,
            stream_url="/video_feed",
            model_path=CURRENT_MODEL_PATH,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
        )

    return index()

# ----------------------------------------------------
# 이미지 결과 서빙
# ----------------------------------------------------
@app.route("/image_result")
def image_result():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "No image", 404
    return send_file(path, mimetype="image/jpeg")

# ----------------------------------------------------
# 영상 스트리밍
# ----------------------------------------------------
def generate_video_stream():
    cap = cv2.VideoCapture(CURRENT_VIDEO_PATH)
    model = load_model(CURRENT_MODEL_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
            verbose=False,
        )

        if NEW_CLASS_NAMES:
            results[0].names = NEW_CLASS_NAMES
            if hasattr(model, "predictor"):
                model.predictor.model.names = NEW_CLASS_NAMES

        plotted = results[0].plot()

        ok, buf = cv2.imencode(".jpg", plotted)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )

    cap.release()

# ----------------------------------------------------
# 영상 스트림 엔드포인트
# ----------------------------------------------------
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# ----------------------------------------------------
# 실행
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4403, debug=False)
