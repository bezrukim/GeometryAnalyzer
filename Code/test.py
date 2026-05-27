from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="annotations/data.yaml")

print(f"mAP@50 (маски): {metrics.box.map50:.3f}")  # для bbox
print(f"mAP@50 (маски): {metrics.seg.map50:.3f}")