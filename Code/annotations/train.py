from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")  # маленькая и быстрая модель

model.train(
    data="annotations/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)