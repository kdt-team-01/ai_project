import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import os
import time

# -----------------------------
# 설정값 및 상수 정의
# -----------------------------
SUPPORTED_MEDIA_TYPES = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
# 모델 파일 경로를 'yolo11n.pt'로 수정했습니다.
DEFAULT_MODEL_PATH = "yolo11n.pt" 

# Flask 앱에서 가져온 클래스 이름 재정의
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
# YOLO 모델 로드 (캐시 사용)
# -----------------------------
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        
        # --- 오류 해결을 위한 클래스 이름 재정의 수정 ---
        if NEW_CLASS_NAMES:
            # model.names 대신 내부 모델 객체의 names 속성을 수정합니다.
            if hasattr(model.model, 'names'):
                 model.model.names = NEW_CLASS_NAMES
                 st.success("클래스 이름 재정의 적용 완료.")
            else:
                 # YOLOv8+ 버전이 아닐 경우를 대비한 대체 로직 (경고만 표시)
                 st.warning("경고: 모델 내부 이름 속성(names)을 찾을 수 없습니다. 이름 재정의가 적용되지 않을 수 있습니다.")
        # ---------------------------------------------
            
        return model
    except Exception as e:
        st.error(f"모델 로드 오류: {e}. 경로와 파일명을 다시 확인해주세요.")
        return None

# -----------------------------
# Streamlit 앱 시작
# -----------------------------

st.title("🚗 YOLO 객체 탐지 간이 테스트 (YOLO11n)")
st.markdown("이미지나 영상을 업로드하여 YOLO 모델의 실시간 탐지 결과를 확인하세요.")

# --- 사이드바에서 설정값 받기 ---
st.sidebar.header("⚙️ 모델 및 추론 설정")
model_path = st.sidebar.text_input(
    "모델 파일 경로 (.pt)", 
    DEFAULT_MODEL_PATH
)
conf_threshold = st.sidebar.slider(
    "Confidence Threshold (확신도)", 
    min_value=0.0, max_value=1.0, 
    value=0.25, step=0.05
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold (겹침 허용치)", 
    min_value=0.0, max_value=1.0, 
    value=0.45, step=0.05
)

# --- 파일 업로드 위젯 ---
uploaded_file = st.file_uploader(
    "이미지 또는 영상 파일 업로드", 
    type=SUPPORTED_MEDIA_TYPES
)

if uploaded_file is not None:
    # 1. 모델 로드
    model = load_yolo_model(model_path)

    if model:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # -----------------------------
        # 이미지 파일 처리
        # -----------------------------
        if file_extension in ["jpg", "jpeg", "png"]:
            st.header("🖼️ 이미지 탐지 결과")
            
            # 파일 스트림을 넘파이 배열로 변환
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1) # OpenCV가 BGR로 디코딩

            # 추론
            results = model.predict(
                source=img, 
                conf=conf_threshold, 
                iou=iou_threshold, 
                verbose=False
            )
            
            plotted_bgr = results[0].plot()
            plotted_rgb = plotted_bgr[:, :, ::-1] # BGR을 RGB로 변환
            
            st.image(plotted_rgb, caption="탐지 결과", use_column_width=True)


        # -----------------------------
        # 영상 파일 처리
        # -----------------------------
        elif file_extension in ["mp4", "avi", "mov", "mkv"]:
            st.header("🎥 영상 탐지 (프레임 단위)")

            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name

            # 스트리밍 처리 (Streamlit의 placeholder 사용)
            video_placeholder = st.empty()
            st_status = st.empty()
            
            cap = cv2.VideoCapture(temp_video_path)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO 추론
                results = model.predict(
                    source=frame, 
                    conf=conf_threshold, 
                    iou=iou_threshold, 
                    verbose=False
                )
                
                plotted_bgr = results[0].plot()
                plotted_rgb = plotted_bgr[:, :, ::-1] # BGR을 RGB로 변환
                
                video_placeholder.image(plotted_rgb, channels="RGB")
                
                frame_count += 1
                st_status.text(f"처리된 프레임 수: {frame_count}")
                
                # Streamlit의 높은 부하를 줄이기 위해 짧게 쉼
                time.sleep(0.01)

            cap.release()
            st_status.success(f"총 {frame_count} 프레임 처리 완료!")

            # 임시 파일 삭제
            os.unlink(temp_video_path)
        
        else:
            st.error("지원하지 않는 파일 형식입니다.")