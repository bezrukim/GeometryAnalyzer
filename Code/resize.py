import cv2
import numpy as np

img = cv2.imread("frame45.jpg")
h, w = img.shape[:2]

with open("frame45.txt") as f:
    for line in f:
        data = list(map(float, line.split()))
        pts = np.array(data[1:]).reshape(-1,2)
        pts[:,0] *= w
        pts[:,1] *= h
        pts = pts.astype(int)

        cv2.polylines(img, [pts], True, (0,255,0), 2)

while True:
    cv2.imshow("check", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break