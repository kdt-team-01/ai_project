import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# -----------------------------
# 설정값 및 상수 정의
# -----------------------------
# 파일 업로드 유형 정의
SUPPORTED_MEDIA_TYPES = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
# 모델 파일 (예시)
DEFAULT_MODEL_PATH = "yolov8n.pt" 

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
# Streamlit 앱 시작
# -----------------------------

st.title("🚗 YOLO 객체 탐지 간이 테스트")
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
    # 1. 모델 로드 (캐싱을 통해 재로드 방지)
    @st.cache_resource
    def load_yolo_model(path):
        try:
            model = YOLO(path)
            # 클래스 이름 재정의 적용
            if NEW_CLASS_NAMES:
                model.names = NEW_CLASS_NAMES
            return model
        except Exception as e:
            st.error(f"모델 로드 오류: {e}")
            return None

    model = load_yolo_model(model_path)

    if model:
        # 2. 파일 타입 확인
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
            
            # 박스 그리기 (BGR 결과)
            plotted_bgr = results[0].plot()
            
            # BGR을 RGB로 변환 (Streamlit은 RGB를 사용)
            plotted_rgb = plotted_bgr[:, :, ::-1]
            
            # 결과 표시
            st.image(plotted_rgb, caption="탐지 결과", use_column_width=True)


        # -----------------------------
        # 영상 파일 처리
        # -----------------------------
        elif file_extension in ["mp4", "avi", "mov", "mkv"]:
            st.header("🎥 영상 탐지 (프레임 단위)")

            # 임시 파일로 저장
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # 스트리밍 처리 (Streamlit의 placeholder 사용)
            video_placeholder = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            
            # 비디오 처리 상태 표시
            st_status = st.empty()

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
                
                # 박스 그리기 (BGR)
                plotted_bgr = results[0].plot()
                
                # BGR을 RGB로 변환 (Streamlit 출력용)
                plotted_rgb = plotted_bgr[:, :, ::-1]

                # Streamlit에 프레임 표시
                video_placeholder.image(plotted_rgb, channels="RGB")
                
                frame_count += 1
                st_status.text(f"처리된 프레임 수: {frame_count}")
                
            cap.release()
            st_status.success("영상 처리 완료!")

            # 임시 파일 삭제
            os.unlink(tfile.name)
        
        else:
            st.error("지원하지 않는 파일 형식입니다.")