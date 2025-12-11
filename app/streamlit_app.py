import streamlit as st
from PIL import Image
import numpy as np
import tempfile

from ultralytics import YOLO

st.set_page_config(page_title="Adverse Weather CCTV YOLO Demo", layout="wide")

st.title("🌧️❄️🌫️ 악천후 CCTV YOLO 데모")
st.caption("이미지/영상 업로드 → 탐지 결과 시현 (MVP)")

# ----------------------------
# 모델 로드 캐시
# ----------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

# 기본 모델(임시)
DEFAULT_MODEL = "yolo11n.pt"  # 안 되면 yolo8n.pt로 변경

# ----------------------------
# 사이드바
# ----------------------------
st.sidebar.header("⚙️ 설정")
model_path = st.sidebar.text_input("모델 경로", value=DEFAULT_MODEL)
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("IoU", 0.1, 1.0, 0.45, 0.05)
mode = st.sidebar.radio("입력 종류", ["이미지", "영상"])

# ----------------------------
# 모델 로드
# ----------------------------
try:
    model = load_model(model_path)
    st.sidebar.success("모델 로드 성공")
except Exception as e:
    st.sidebar.error(f"모델 로드 실패: {e}")
    st.stop()

# ----------------------------
# 영상 모드 (전체 영상에 박스 씌운 결과 mp4 생성)
# ----------------------------
else:
    uploaded = st.file_uploader("영상 업로드", type=["mp4", "avi", "mov", "mkv"])

    # 성능 옵션
    st.sidebar.subheader("🎬 영상 옵션")
    frame_skip = st.sidebar.slider("프레임 스킵(속도용)", 1, 10, 2, 1)  # 1이면 모든 프레임
    resize_w = st.sidebar.selectbox("리사이즈 폭(속도용)", [None, 1280, 960, 720, 640], index=2)

    if uploaded:
        # 원본 저장
        in_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        in_file.write(uploaded.read())
        video_path = in_file.name

        st.subheader("원본 영상")
        st.video(video_path)

        if st.button("🚀 영상 전체 탐지해서 결과 영상 만들기"):
            import cv2
            import os

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("영상 파일을 열 수 없습니다.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 리사이즈 적용
            if resize_w is not None and resize_w < w:
                scale = resize_w / w
                out_w = int(w * scale)
                out_h = int(h * scale)
            else:
                out_w, out_h = w, h

            # 출력 파일
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 20, (out_w, out_h))

            progress = st.progress(0)
            status = st.empty()

            idx = 0
            processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임 스킵
                if idx % frame_skip != 0:
                    idx += 1
                    continue

                # 리사이즈
                if (out_w, out_h) != (w, h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # YOLO 추론
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )

                # 박스 그려진 프레임
                plotted = results[0].plot()  # BGR

                writer.write(plotted)

                processed += 1
                idx += 1

                # 진행 표시
                if total > 0:
                    progress_val = min(1.0, idx / total)
                    progress.progress(progress_val)
                    status.write(f"처리 중... {idx}/{total} 프레임")

            cap.release()
            writer.release()
            progress.progress(1.0)
            status.write("✅ 변환 완료!")

            st.subheader("✅ 탐지 결과 영상")
            st.video(out_path)

            st.info(
                "※ Streamlit은 원본 영상 위에 실시간 오버레이가 어려워서 "
                "탐지된 새 영상을 만들어 재생하는 방식이 가장 안정적입니다."
            )