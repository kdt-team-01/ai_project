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
# 이미지 모드
# ----------------------------
if mode == "이미지":
    uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("원본")
            st.image(img, use_container_width=True)

        results = model.predict(
            source=np.array(img),
            conf=conf,
            iou=iou,
            verbose=False
        )

        plotted = results[0].plot()  # BGR
        plotted = plotted[:, :, ::-1]  # RGB 변환

        with col2:
            st.subheader("탐지 결과")
            st.image(plotted, use_container_width=True)

        st.info(f"탐지 객체 수: {len(results[0].boxes)}")

else:
    uploaded = st.file_uploader("영상 업로드", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        video_path = tfile.name

        st.video(video_path)

        import cv2

        st.sidebar.subheader("🎞️ 영상 옵션")
        frame_skip = st.sidebar.slider("프레임 간격(클수록 빠름)", 1, 30, 5)

        if st.button("영상 감지 실행"):
            cap = cv2.VideoCapture(video_path)

            view = st.empty()
            idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                idx += 1
                if idx % frame_skip != 0:
                    continue

                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )

                plotted = results[0].plot()  # BGR
                plotted = plotted[:, :, ::-1]  # RGB

                view.image(plotted, use_container_width=True)

            cap.release()
            st.success("영상 감지 완료!")
