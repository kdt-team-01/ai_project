from flask import Flask, render_template, request, Response, send_file, redirect, url_for
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

from urllib.parse import urlparse
import ipaddress
import socket
import subprocess
import shutil
from typing import Optional, Dict

# ----------------------------------------------------
# PyInstaller 대응 (templates/static 경로 고정)
# ----------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, "templates"),
    static_folder=os.path.join(base_dir, "static"),
)

# ----------------------------------------------------
# 전역 상태
# ----------------------------------------------------
CURRENT_VIDEO_PATH: Optional[str] = None
CURRENT_SOURCE_URL: Optional[str] = None
CURRENT_SOURCE_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.utic.go.kr/",
}

CURRENT_MODEL_PATH = "best.pt"   # 기본은 너 모델
CURRENT_CONF = 0.25
CURRENT_IOU = 0.45

CURRENT_MODEL = None  # YOLO 모델 캐시

# ----------------------------------------------------
# best.pt(너 커스텀) 라벨
# ----------------------------------------------------
BEST_KR = {
    0: "승용차",
    1: "소형버스",
    2: "대형버스",
    3: "트럭",
    4: "대형트레일러",
    5: "오토바이",
    6: "보행자",
}

# ----------------------------------------------------
# COCO(80) 모델(yolo11n/m 등) 일부 한글 라벨
#  - 전부 번역할 필요 없고, 필요한 것만 덮어쓰기
# ----------------------------------------------------
COCO_KR = {
    0: "사람",       # person
    2: "승용차",     # car
    3: "오토바이",   # motorcycle
    5: "버스",       # bus
    7: "트럭",       # truck
}

# ----------------------------------------------------
# (선택) 캐시 방지: 템플릿/화면 수정 바로 반영
# ----------------------------------------------------
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp

# ----------------------------------------------------
# YOLO 모델 로드(캐시)
# ----------------------------------------------------
def load_model(path: str):
    global CURRENT_MODEL
    if CURRENT_MODEL is None or getattr(CURRENT_MODEL, "model_path", None) != path:
        CURRENT_MODEL = YOLO(path)
        CURRENT_MODEL.model_path = path
    return CURRENT_MODEL

# ----------------------------------------------------
# ✅ names 안전 적용 (KeyError 방지 핵심)
#  - 기존 model.names 전체를 유지한 채로 일부만 덮어씀
#  - best.pt면 BEST_KR, 그 외는 COCO_KR
# ----------------------------------------------------
def apply_names(result0, model, model_path: str):
    base = model.names

    # ultralytics에서 names가 list일 수도 있어서 dict로 변환
    if isinstance(base, (list, tuple)):
        base = {i: n for i, n in enumerate(base)}
    elif not isinstance(base, dict):
        base = {}

    merged = dict(base)  # ✅ 전체 클래스 유지(중요)

    if os.path.basename(model_path).lower() == "best.pt":
        merged.update(BEST_KR)
    else:
        merged.update(COCO_KR)

    # plot()은 보통 result0.names를 쓰지만, 안전하게 둘 다 세팅
    result0.names = merged
    try:
        if hasattr(model, "predictor") and model.predictor and hasattr(model.predictor, "model"):
            model.predictor.model.names = merged
    except Exception:
        pass

# ----------------------------------------------------
# SSRF 방지: 내부망/로컬 IP 차단
# ----------------------------------------------------
def is_public_host(hostname: str) -> bool:
    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            return False
        return True
    except Exception:
        return False

def validate_stream_url(u: str) -> Optional[str]:
    u = (u or "").strip()
    if not u:
        return None
    p = urlparse(u)
    if p.scheme not in ("http", "https", "rtsp"):
        return None
    if not p.hostname:
        return None
    if not is_public_host(p.hostname):
        return None
    return u

# ----------------------------------------------------
# 공통 렌더
# ----------------------------------------------------
def render_home(image_url=None, stream_url=None):
    return render_template(
        "index.html",
        image_url=image_url,
        stream_url=stream_url,
        model_path=CURRENT_MODEL_PATH,
        conf=CURRENT_CONF,
        iou=CURRENT_IOU,
        source_url=CURRENT_SOURCE_URL or "",
        referer=CURRENT_SOURCE_HEADERS.get("Referer", ""),
        user_agent=CURRENT_SOURCE_HEADERS.get("User-Agent", ""),
    )

# ----------------------------------------------------
# 메인
# ----------------------------------------------------
@app.route("/")
def index():
    stream_url = "/video_feed" if (CURRENT_VIDEO_PATH or CURRENT_SOURCE_URL) else None
    return render_home(image_url=None, stream_url=stream_url)

# ----------------------------------------------------
# URL 소스 설정 (PRG: POST 후 redirect로 /로 복귀)
# ----------------------------------------------------
@app.route("/set_url", methods=["GET", "POST"])
def set_url():
    global CURRENT_SOURCE_URL, CURRENT_VIDEO_PATH, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU, CURRENT_SOURCE_HEADERS

    if request.method == "GET":
        return redirect(url_for("index"))

    # 설정값 반영
    CURRENT_MODEL_PATH = request.form.get("model_path", CURRENT_MODEL_PATH)
    CURRENT_CONF = float(request.form.get("conf", CURRENT_CONF))
    CURRENT_IOU = float(request.form.get("iou", CURRENT_IOU))

    url = validate_stream_url(request.form.get("source_url", ""))
    if not url:
        CURRENT_SOURCE_URL = None
        CURRENT_VIDEO_PATH = None
        return redirect(url_for("index"))

    referer = (request.form.get("referer") or "https://www.utic.go.kr/").strip()
    user_agent = (request.form.get("user_agent") or "Mozilla/5.0").strip()
    CURRENT_SOURCE_HEADERS = {"Referer": referer, "User-Agent": user_agent}

    CURRENT_SOURCE_URL = url
    CURRENT_VIDEO_PATH = None
    return redirect(url_for("index"))

# ----------------------------------------------------
# 업로드 처리
# ----------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    global CURRENT_VIDEO_PATH, CURRENT_SOURCE_URL, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU

    f = request.files.get("file")

    CURRENT_MODEL_PATH = request.form.get("model_path", CURRENT_MODEL_PATH)
    CURRENT_CONF = float(request.form.get("conf", CURRENT_CONF))
    CURRENT_IOU = float(request.form.get("iou", CURRENT_IOU))

    if not f:
        return redirect(url_for("index"))

    filename = (f.filename or "").lower()

    # 업로드 모드로 전환 (URL 모드 해제)
    CURRENT_SOURCE_URL = None

    # ---------- 이미지 ----------
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_pil = Image.open(f.stream).convert("RGB")
        img = np.array(img_pil)  # RGB

        # 너무 큰 사진이면 리사이즈
        h, w = img.shape[:2]
        max_w = 1280
        if w > max_w:
            ratio = max_w / w
            img = cv2.resize(img, (max_w, int(h * ratio)))

        model = load_model(CURRENT_MODEL_PATH)

        with torch.no_grad():
            results = model.predict(
                source=img,
                conf=CURRENT_CONF,
                iou=CURRENT_IOU,
                imgsz=1280,
                verbose=False,
            )

        apply_names(results[0], model, CURRENT_MODEL_PATH)

        try:
            plotted = results[0].plot()  # BGR
        except Exception:
            plotted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.fromarray(plotted).save(out.name)

        return render_home(image_url="/image_result?path=" + out.name, stream_url=None)

    # ---------- 영상 ----------
    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.save(tfile.name)
        CURRENT_VIDEO_PATH = tfile.name
        return redirect(url_for("index"))

    return redirect(url_for("index"))

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
# FFmpeg fallback (URL/HLS 등 OpenCV가 못 열 때)
# ----------------------------------------------------
def ffmpeg_frames(url: str, headers: Dict[str, str]):
    if shutil.which("ffmpeg") is None:
        return None

    header_lines = ""
    for k, v in (headers or {}).items():
        header_lines += f"{k}: {v}\r\n"

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if header_lines:
        cmd += ["-headers", header_lines]
    cmd += [
        "-i", url,
        "-vf", "scale=960:-1",
        "-an",
        "-f", "mjpeg",
        "pipe:1",
    ]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buf = b""
    try:
        while True:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf += chunk

            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = buf[start:end + 2]
                buf = buf[end + 2:]

                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
    finally:
        try:
            p.kill()
        except Exception:
            pass

# ----------------------------------------------------
# 영상 스트리밍
# ----------------------------------------------------
def generate_video_stream():
    source = CURRENT_SOURCE_URL if CURRENT_SOURCE_URL else CURRENT_VIDEO_PATH
    if not source:
        return

    model = load_model(CURRENT_MODEL_PATH)

    cap = cv2.VideoCapture(source)
    use_ffmpeg = not cap.isOpened()

    frame_iter = None
    if use_ffmpeg:
        frame_iter = ffmpeg_frames(source, CURRENT_SOURCE_HEADERS)
        if frame_iter is None:
            return

    try:
        while True:
            if use_ffmpeg:
                try:
                    frame = next(frame_iter)
                except StopIteration:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # 너무 줄이면 탐지 더 약해지니까 1280 유지
            h, w = frame.shape[:2]
            max_w = 1280
            if w > max_w:
                ratio = max_w / w
                frame = cv2.resize(frame, (max_w, int(h * ratio)))

            plotted = frame
            try:
                with torch.no_grad():
                    results = model.predict(
                        source=frame,
                        conf=CURRENT_CONF,
                        iou=CURRENT_IOU,
                        imgsz=1280,
                        verbose=False,
                    )

                apply_names(results[0], model, CURRENT_MODEL_PATH)
                plotted = results[0].plot()  # BGR
            except Exception:
                plotted = frame  # 에러 나도 영상은 계속

            ok, buf = cv2.imencode(".jpg", plotted)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    finally:
        if cap and cap.isOpened():
            cap.release()

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
    print("RUNNING FILE:", __file__)
    print(app.url_map)
    port = int(os.environ.get("PORT", "5001"))  # 5000 꼬이면 5001
    app.run(host="0.0.0.0", port=port, debug=False)
