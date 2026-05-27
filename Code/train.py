from ultralytics import YOLO

model = YOLO("yolov8l-seg.pt")  # маленькая и быстрая модель

model.train(
    data="C:\\Users\\bezrukim\\Documents\\Учёба\\Курсовая\\Code\\annotations\\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)