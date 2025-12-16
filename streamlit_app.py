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
# 모델 파일 경로: 요청하신 'yolo11n.pt' 유지
DEFAULT_MODEL_PATH = "yolo11n.pt" 

# 클래스 이름 재정의
NEW_CLASS_NAMES = {
    0: "승용차",
    1: "소형버스",
    2: "대형버스",
    3: "트럭",
    4: "대형트레일러",
    5: "오토바이",
    6: "보행자",
    # 참고: 만약 모델이 감지하는 클래스 ID가 7번 이상이라면
    # 여기에 추가적인 클래스 이름을 정의해주어야 KeyError가 발생하지 않습니다.
}

# -----------------------------
# YOLO 모델 로드 (캐시 사용)
# -----------------------------
@st.cache_resource
def load_yolo_model(path):
    """모델 경로가 바뀌면 다시 로드, 아니면 기존 모델 재사용"""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"모델 로드 오류: {e}. 경로와 파일명을 다시 확인해주세요.")
        # 모델 파일(yolo11n.pt)이 GitHub 저장소 루트에 있는지 확인해주세요.
        return None

# -----------------------------
# Streamlit 앱 시작
# -----------------------------

st.set_page_config(layout="wide")
st.title("🚗 YOLO 객체 탐지 간이 테스트 (YOLO11n)")
st.markdown("이미지나 영상을 업로드하여 YOLO 모델의 추론 결과를 확인하세요.")

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

# 파일이 업로드되면 처리 시작
if uploaded_file is not None:
    # 1. 모델 로드
    model = load_yolo_model(model_path)

    if model:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        # -----------------------------
        # 추론 및 플롯 공통 함수 (KeyError 방지 로직 포함)
        # -----------------------------
        def process_and_plot(source_data, frame_num=None):
            """YOLO 추론을 수행하고 플롯된 BGR 이미지를 반환"""
            
            # 추론
            results = model.predict(
                source=source_data, 
                conf=conf_threshold, 
                iou=iou_threshold, 
                verbose=False
            )
            
            # *** KeyError 방지를 위한 클래스 이름 재정의 및 필터링 ***
            if results and results[0]:
                results[0].names = NEW_CLASS_NAMES
                
                # 감지된 객체 ID가 NEW_CLASS_NAMES 범위를 벗어날 경우 필터링
                if results[0].boxes is not None:
                    # 감지된 모든 클래스 ID를 가져와서 NEW_CLASS_NAMES의 키와 비교
                    valid_indices = [i for i, c in enumerate(results[0].boxes.cls.tolist()) if int(c) in NEW_CLASS_NAMES]
                    
                    if valid_indices:
                        # 유효한 객체만 남기기
                        results[0].boxes = results[0].boxes[valid_indices]
                        # 마스크, 키포인트 등 다른 결과도 필터링할 수 있으나, 여기서는 박스만 처리
                    else:
                        if frame_num is not None:
                             st.warning(f"경고: 프레임 {frame_num}에서 유효한 객체가 감지되지 않았습니다. (클래스 ID 불일치)")
                        else:
                             st.warning("경고: 이미지에서 유효한 객체가 감지되지 않았습니다. (클래스 ID 불일치)")
                        # 유효한 객체가 없어도 빈 이미지를 플롯할 수 있도록 results 객체는 유지

            plotted_bgr = results[0].plot()
            return plotted_bgr


        # -----------------------------
        # 이미지 파일 처리
        # -----------------------------
        if file_extension in ["jpg", "jpeg", "png"]:
            st.header("🖼️ 이미지 탐지 결과")
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1) # BGR로 디코딩
            
            plotted_bgr = process_and_plot(img)
            plotted_rgb = plotted_bgr[:, :, ::-1] # BGR을 RGB로 변환
            
            st.image(plotted_rgb, caption="탐지 결과", use_column_width=True)


        # -----------------------------
        # 영상 파일 처리 (스크롤 기반 탐색)
        # -----------------------------
        elif file_extension in ["mp4", "avi", "mov", "mkv"]:
            st.header("🎥 영상 탐지 (프레임 탐색 모드)")

            # 임시 파일로 저장 (st.cache_data를 사용하지 않으므로 임시 파일 필요)
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name

            cap = cv2.VideoCapture(temp_video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.info(f"총 프레임 수: {total_frames} | FPS: {fps:.2f}")

            # 

            # 2. 프레임 슬라이더 위젯 (볼륨 컨트롤 바와 같은 역할)
            frame_number = st.slider(
                "프레임 번호 선택", 
                min_value=0, 
                max_value=total_frames - 1, 
                value=0, 
                step=1
            )
            
            # 3. 선택된 프레임 위치로 이동 및 읽기
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # 4. YOLO 추론 및 플롯
                plotted_bgr = process_and_plot(frame, frame_number)
                plotted_rgb = plotted_bgr[:, :, ::-1] # BGR을 RGB로 변환
                
                # 5. 결과 이미지 표시
                st.image(plotted_rgb, caption=f"프레임 {frame_number} 탐지 결과", use_column_width=True)
                
            else:
                st.error("선택된 프레임을 읽는 데 실패했습니다. 파일을 다시 업로드해 주세요.")
                
            cap.release()
            os.unlink(temp_video_path) # 임시 파일 삭제
        
        else:
            st.error("지원하지 않는 파일 형식입니다.")