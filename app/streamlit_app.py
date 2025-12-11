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
# 영상 모드 (프레임 스트리밍 방식)
# ----------------------------
else:
    uploaded = st.file_uploader("영상 업로드", type=["mp4", "avi", "mov", "mkv"])

    st.sidebar.subheader("🎬 영상 옵션")
    frame_skip = st.sidebar.slider("프레임 스킵(속도용)", 0, 10, 2, 1)  
    # 0이면 매 프레임 추론, 2면 3프레임 중 1프레임 추론 느낌
    max_width = st.sidebar.selectbox("리사이즈 폭(속도용)", [640, 800, 960, 1280], index=2)
    play_fps = st.sidebar.slider("표시 FPS(느낌)", 1, 30, 12, 1)

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        video_path = tfile.name

        st.info("✅ 아래 영역에서 업로드 영상이 '탐지 오버레이된 형태로' 바로 재생처럼 표시됩니다.")
        st.caption("※ Streamlit 기본 플레이어 위 실시간 오버레이는 어려워서, 프레임을 연속 출력하는 방식입니다.")

        # 재생 제어용 상태
        if "playing" not in st.session_state:
            st.session_state.playing = False

        colA, colB = st.columns(2)
        with colA:
            if st.button("▶️ 재생", use_container_width=True):
                st.session_state.playing = True
        with colB:
            if st.button("⏸️ 정지", use_container_width=True):
                st.session_state.playing = False

        display_area = st.empty()
        progress = st.progress(0)

        import cv2
        import time

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

        idx = 0
        last_time = time.time()

        # 재생 루프
        while cap.isOpened() and st.session_state.playing:
            ret, frame = cap.read()
            if not ret:
                break

            # 진행률
            if total > 0:
                progress.progress(min(idx / total, 1.0))

            # 리사이즈(속도)
            h, w = frame.shape[:2]
            if w > max_width:
                new_h = int(h * (max_width / w))
                frame = cv2.resize(frame, (max_width, new_h))

            # 프레임 스킵 기반 추론
            if frame_skip == 0 or (idx % (frame_skip + 1) == 0):
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )
                plotted = results[0].plot()  # BGR
            else:
                plotted = frame

            # BGR -> RGB
            plotted_rgb = plotted[:, :, ::-1]

            # 화면 표시(영상처럼)
            display_area.image(plotted_rgb, use_container_width=True)

            idx += 1

            # 표시 FPS 조절(느낌)
            elapsed = time.time() - last_time
            target_delay = max(1.0 / play_fps - elapsed, 0)
            time.sleep(target_delay)
            last_time = time.time()

        cap.release()
        progress.empty()

        if not st.session_state.playing:
            st.warning("⏸️ 정지 상태입니다. 재생을 누르면 다시 시작합니다.")
        else:
            st.success("✅ 영상 끝까지 재생 완료!")