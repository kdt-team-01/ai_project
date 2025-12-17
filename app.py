from flask import Flask, render_template, request, Response, send_file, redirect, url_for
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
<<<<<<< Updated upstream
=======
import torch

from urllib.parse import urlparse
import ipaddress
import socket
import subprocess
import shutil
from typing import Optional, Dict
>>>>>>> Stashed changes

# ----------------------------------------------------
# PyInstaller 대응
# ----------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

<<<<<<< Updated upstream
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
=======
# =========================
# 전역 상태
# =========================
CURRENT_VIDEO_PATH: Optional[str] = None
CURRENT_SOURCE_URL: Optional[str] = None

CURRENT_SOURCE_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.utic.go.kr/",
}

CURRENT_MODEL_PATH = "yolo11n.pt"
>>>>>>> Stashed changes
CURRENT_CONF = 0.25
CURRENT_IOU = 0.45
CURRENT_MODEL = None

<<<<<<< Updated upstream
# ----------------------------------------------------
# 클래스 이름 재정의
# ----------------------------------------------------
=======
# =========================
# YOLO 모델 로드 (간단 캐시)
# =========================
def load_model(path: str):
    global CURRENT_MODEL
    if "CURRENT_MODEL" not in globals() or getattr(CURRENT_MODEL, "model_path", None) != path:
        CURRENT_MODEL = YOLO(path)
        CURRENT_MODEL.model_path = path
    return CURRENT_MODEL

# =========================
# 너 커스텀(best.pt) 라벨
# =========================
>>>>>>> Stashed changes
NEW_CLASS_NAMES = {
    0: "승용차",
    1: "소형버스",
    2: "대형버스",
    3: "트럭",
    4: "대형트레일러",
    5: "오토바이",
    6: "보행자",
}

<<<<<<< Updated upstream
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
=======
# =========================
# COCO(80클래스)용 일부 한글 라벨(필요한 것만)
# =========================
COCO_KR = {
    0: "사람",       # person
    2: "승용차",     # car
    3: "오토바이",   # motorcycle
    5: "버스",       # bus
    7: "트럭",       # truck
}

# =========================
# ✅ names 안전 적용 (KeyError 방지 + best.pt 라벨 꼬임 방지)
# =========================
def apply_names(result0, model, model_path: str):
    base = model.names
    if isinstance(base, (list, tuple)):
        base = {i: n for i, n in enumerate(base)}
    merged = dict(base)  # ✅ 전체 클래스 유지

    # ✅ best.pt면 무조건 너 커스텀 라벨 적용
    if os.path.basename(model_path).lower() == "best.pt":
        merged.update(NEW_CLASS_NAMES)
    else:
        # ✅ yolo11n/m 같은 COCO 모델이면 일부만 한글 덮어쓰기
        merged.update(COCO_KR)

    result0.names = merged

# =========================
# (선택) 캐시 방지
# =========================
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp

# =========================
# SSRF 방지: 내부망/로컬 IP 차단
# =========================
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

# =========================
# 공통 렌더
# =========================
def render_home(image_url=None, stream_url=None):
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
# =========================
# 메인
# =========================
@app.route("/")
def index():
    stream_url = "/video_feed" if (CURRENT_VIDEO_PATH or CURRENT_SOURCE_URL) else None
    return render_home(image_url=None, stream_url=stream_url)

# =========================
# URL 소스 설정 (PRG)
# =========================
@app.route("/set_url", methods=["GET", "POST"])
def set_url():
    global CURRENT_SOURCE_URL, CURRENT_VIDEO_PATH, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU, CURRENT_SOURCE_HEADERS

    if request.method == "GET":
        return redirect(url_for("index"))

    CURRENT_MODEL_PATH = request.form.get("model_path", "yolo11n.pt")
    CURRENT_CONF = float(request.form.get("conf", 0.25))
    CURRENT_IOU = float(request.form.get("iou", 0.45))

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

# =========================
# 업로드 처리
# =========================
@app.route("/upload", methods=["POST"])
def upload():
    global CURRENT_VIDEO_PATH, CURRENT_SOURCE_URL, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU

    f = request.files.get("file")

    CURRENT_MODEL_PATH = request.form.get("model_path", "yolo11n.pt")
    CURRENT_CONF = float(request.form.get("conf", 0.25))
    CURRENT_IOU = float(request.form.get("iou", 0.45))

    if not f:
        return redirect(url_for("index"))

    filename = (f.filename or "").lower()

    # 업로드 모드로 전환
    CURRENT_SOURCE_URL = None

    # ---------- 이미지 ----------
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_pil = Image.open(f.stream).convert("RGB")
        img = np.array(img_pil)

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
            plotted = results[0].plot()
        except Exception:
            plotted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
>>>>>>> Stashed changes

        out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.fromarray(plotted).save(out.name)

<<<<<<< Updated upstream
        return render_template(
            "index.html",
            image_url="/image_result?path=" + out.name,
            stream_url=None,
            model_path=CURRENT_MODEL_PATH,
            conf=CURRENT_CONF,
            iou=CURRENT_IOU,
        )
=======
        return render_home(image_url="/image_result?path=" + out.name, stream_url=None)
>>>>>>> Stashed changes

    # ---------- 영상 ----------
    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.save(tfile.name)
        CURRENT_VIDEO_PATH = tfile.name
<<<<<<< Updated upstream

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
=======
        return redirect(url_for("index"))

    return redirect(url_for("index"))

# =========================
# 이미지 결과 서빙
# =========================
>>>>>>> Stashed changes
@app.route("/image_result")
def image_result():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "No image", 404
    return send_file(path, mimetype="image/jpeg")

<<<<<<< Updated upstream
# ----------------------------------------------------
# 영상 스트리밍
# ----------------------------------------------------
def generate_video_stream():
    cap = cv2.VideoCapture(CURRENT_VIDEO_PATH)
=======
# =========================
# FFmpeg fallback (URL/HLS 등 OpenCV가 못 열 때)
# =========================
def ffmpeg_frames(url: str, headers: Dict[str, str]):
    if shutil.which("ffmpeg") is None:
        return

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
                jpg = buf[start:end+2]
                buf = buf[end+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
    finally:
        try:
            p.kill()
        except Exception:
            pass

# =========================
# 영상 스트리밍
# =========================
def generate_video_stream():
    source = CURRENT_SOURCE_URL if CURRENT_SOURCE_URL else CURRENT_VIDEO_PATH
    if not source:
        return

>>>>>>> Stashed changes
    model = load_model(CURRENT_MODEL_PATH)

    cap = cv2.VideoCapture(source)
    use_ffmpeg = not cap.isOpened()

<<<<<<< Updated upstream
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
=======
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

            # 리사이즈(너무 줄이면 탐지 더 안 됨)
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

                plotted = results[0].plot()
            except Exception:
                plotted = frame  # 실패해도 영상은 계속 나가게

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

>>>>>>> Stashed changes
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

<<<<<<< Updated upstream
# ----------------------------------------------------
# 실행
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4403, debug=False)
=======
if __name__ == "__main__":
    print("RUNNING FILE:", __file__)
    print(app.url_map)

    # 5000 꼬이면 5001 고정
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=False)
>>>>>>> Stashed changes
