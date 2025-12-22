from flask import Flask, render_template, request, Response, send_file, redirect, url_for, jsonify
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
# PyInstaller 대응 및 Flask 설정
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

CURRENT_MODEL_PATH = "best.pt"
CURRENT_CONF = 0.25
CURRENT_IOU = 0.45
CURRENT_MODEL = None 

# [추가] 실시간 탐지 개수를 저장할 변수
LATEST_COUNTS = {}

# 라벨 설정 (기존과 동일)
BEST_KR = {0: "승용차", 1: "소형버스", 2: "대형버스", 3: "트럭", 4: "대형트레일러", 5: "오토바이", 6: "보행자"}
COCO_KR = {0: "사람", 2: "승용차", 3: "오토바이", 5: "버스", 7: "트럭"}

# ----------------------------------------------------
# 헬퍼 함수
# ----------------------------------------------------
def load_model(path: str):
    global CURRENT_MODEL
    if CURRENT_MODEL is None or getattr(CURRENT_MODEL, "model_path", None) != path:
        CURRENT_MODEL = YOLO(path)
        CURRENT_MODEL.model_path = path
    return CURRENT_MODEL

def apply_names(result0, model, model_path: str):
    base = model.names
    if isinstance(base, (list, tuple)):
        base = {i: n for i, n in enumerate(base)}
    merged = dict(base if isinstance(base, dict) else {})
    if os.path.basename(model_path).lower() == "best.pt":
        merged.update(BEST_KR)
    else:
        merged.update(COCO_KR)
    result0.names = merged

# ----------------------------------------------------
# SSRF 및 유틸리티 (기존과 동일)
# ----------------------------------------------------
def is_public_host(hostname: str) -> bool:
    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        return not (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local)
    except: return False

def validate_stream_url(u: str) -> Optional[str]:
    u = (u or "").strip()
    if not u: return None
    p = urlparse(u)
    if p.scheme in ("http", "https", "rtsp") and p.hostname and is_public_host(p.hostname):
        return u
    return None

# ----------------------------------------------------
# 라우트 및 API
# ----------------------------------------------------
@app.route("/")
def index():
    stream_url = "/video_feed" if (CURRENT_VIDEO_PATH or CURRENT_SOURCE_URL) else None
    return render_template(
        "index.html",
        stream_url=stream_url,
        model_path=CURRENT_MODEL_PATH,
        conf=CURRENT_CONF,
        iou=CURRENT_IOU,
        source_url=CURRENT_SOURCE_URL or ""
    )

# [추가] 자바스크립트가 호출할 탐지 개수 API
@app.route("/get_counts")
def get_counts():
    # 보행자나 오토바이가 있는지 체크 (알람용)
    has_danger = any(k in LATEST_COUNTS for k in ["보행자", "사람", "오토바이"])
    return jsonify({
        "counts": LATEST_COUNTS,
        "has_danger": has_danger
    })

@app.route("/set_url", methods=["POST"])
def set_url():
    global CURRENT_SOURCE_URL, CURRENT_VIDEO_PATH, CURRENT_MODEL_PATH, CURRENT_CONF, CURRENT_IOU, LATEST_COUNTS
    LATEST_COUNTS = {} # 초기화
    CURRENT_MODEL_PATH = request.form.get("model_path", CURRENT_MODEL_PATH)
    CURRENT_CONF = float(request.form.get("conf", CURRENT_CONF))
    CURRENT_IOU = float(request.form.get("iou", CURRENT_IOU))
    url = validate_stream_url(request.form.get("source_url", ""))
    CURRENT_SOURCE_URL = url
    CURRENT_VIDEO_PATH = None
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload():
    global CURRENT_VIDEO_PATH, CURRENT_SOURCE_URL, LATEST_COUNTS
    f = request.files.get("file")
    if not f:
        return redirect(url_for("index"))
    
    filename = f.filename.lower()
    CURRENT_SOURCE_URL = None
    LATEST_COUNTS = {}

    if filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            # 1. 파일을 바이트로 읽어서 OpenCV 포맷으로 바로 변환 (가장 확실함)
            file_bytes = np.fromstring(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # 2. 모델 예측
            model = load_model(CURRENT_MODEL_PATH)
            results = model.predict(source=img, conf=CURRENT_CONF, iou=CURRENT_IOU, imgsz=1280)
            apply_names(results[0], model, CURRENT_MODEL_PATH)
            
            # 3. 탐지 개수 업데이트
            new_counts = {}
            if results[0].boxes is not None:
                for cls_id in results[0].boxes.cls.cpu().numpy():
                    name = results[0].names.get(int(cls_id), "Unknown")
                    new_counts[name] = new_counts.get(name, 0) + 1
            LATEST_COUNTS = new_counts

            # 4. [색상 문제 종결]
            # YOLO plot() 결과물을 가져옵니다. 
            plotted_img = results[0].plot() 

            # 5. PIL(Image.fromarray)을 쓰지 않고, OpenCV의 imwrite를 사용합니다.
            # OpenCV는 BGR을 표준으로 저장하므로, plotted_img가 BGR이라면 
            # 변환 없이 그대로 저장해야 색깔이 완벽하게 나옵니다.
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(out.name, plotted_img) # 이 한 줄이 정답입니다.
            
            return render_template(
                "index.html", 
                image_url="/image_result?path=" + out.name, 
                model_path=CURRENT_MODEL_PATH,
                conf=CURRENT_CONF,
                iou=CURRENT_IOU,
                source_url=""
            )
        except Exception as e:
            print(f"!!! ERROR !!! : {e}")
            return f"Error: {e}", 500

    # 영상 처리 (생략 없이 그대로 유지)
    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.save(tfile.name)
        CURRENT_VIDEO_PATH = tfile.name
        return redirect(url_for("index"))

    return redirect(url_for("index"))

@app.route("/image_result")
def image_result():
    path = request.args.get("path")
    return send_file(path, mimetype="image/jpeg") if path and os.path.exists(path) else ("No image", 404)

# ----------------------------------------------------
# 영상 스트리밍 (핵심 수정 부분)
# ----------------------------------------------------
def generate_video_stream():
    global LATEST_COUNTS
    source = CURRENT_SOURCE_URL if CURRENT_SOURCE_URL else CURRENT_VIDEO_PATH
    if not source: return

    model = load_model(CURRENT_MODEL_PATH)
    cap = cv2.VideoCapture(source)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 예측
            with torch.no_grad():
                results = model.predict(source=frame, conf=CURRENT_CONF, iou=CURRENT_IOU, imgsz=640, verbose=False)
            
            apply_names(results[0], model, CURRENT_MODEL_PATH)
            
            # [추가] 탐지 개수 실시간 집계
            temp_counts = {}
            for cls_id in results[0].boxes.cls.cpu().numpy():
                name = results[0].names.get(int(cls_id), "Unknown")
                temp_counts[name] = temp_counts.get(name, 0) + 1
            LATEST_COUNTS = temp_counts  # 전역 변수 갱신

            plotted = results[0].plot()
            ok, buf = cv2.imencode(".jpg", plotted)
            if not ok: continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        if cap: cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4403, debug=False)