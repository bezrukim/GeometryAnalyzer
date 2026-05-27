import cv2
import os

cap = cv2.VideoCapture(0)
i = 114
folder = "frame\\"

while True:
    ret, frame = cap.read()
    if not ret or i == 200:
        break

    cv2.imshow("Pencil Geometry Analyzer", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 32: 
        filepath = os.path.join(folder, f"frame{i}.jpg")
        if cv2.imwrite(filepath, frame):
            print(f"Успешное сохранение frame{i}.jpg")
        else:
            print("Не сохранено")
        i += 1

    if key == 27:
        break