from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("runs/detect/train3/weights/last.pt")

# 훈련 코드
def train_yolo_model():
    model.train(
        data="datasets/data.yaml",
        resume=True
    )

# 2. 메인 실행 블록 추가 (필수 수정)
if __name__ == '__main__':
    train_yolo_model()
    # 또는 함수 호출 없이 바로 훈련 코드를 여기에 넣어도 됩니다.
    # model.train(...)