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

# ----------------------------
# 영상 모드 (샘플 프레임만)
# ----------------------------
else:
    uploaded = st.file_uploader("영상 업로드", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        video_path = tfile.name

        st.video(video_path)
        st.warning("영상은 무거울 수 있어 샘플 프레임만 추론합니다.")

        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pick_indices = [
            0,
            frame_count // 3,
            (frame_count * 2) // 3,
            max(0, frame_count - 1)
        ]
        pick_indices = sorted(list(set([i for i in pick_indices if i >= 0])))

        frames_show = []
        idx = 0
        pick_set = set(pick_indices)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in pick_set:
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )
                plotted = results[0].plot()  # BGR
                frames_show.append(plotted)
            idx += 1

        cap.release()

        st.subheader("📌 샘플 프레임 탐지 결과")
        if frames_show:
            for f in frames_show:
                st.image(f[:, :, ::-1], use_container_width=True)
        else:
            st.info("샘플 프레임을 표시하지 못했습니다.")
