import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

def put_text_pil(img, text, pos, font_size=18, color=(255, 255, 255), font_path=None): #Функция вставки текста библиотекой Pillow
    if font_path is None:
        import platform
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/arial.ttf"
        elif platform.system() == "Linux":
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        else:
            font_path = "/System/Library/Fonts/Helvetica.ttc"
            
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
] #Массив цветов разметки

TARGET_H = 480 #Высота панели параметров
PANEL_W = 320 #Ширина панели параметров
prev_time = time.time()
prev_metrics = {} 
ALPHA = 0.35 #Коэффициент сглаживания

model = YOLO("best.pt")
cap = cv2.VideoCapture(0) #Захват видеокамеры
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0 #Вычисление FPS
    prev_time = curr_time

    h, w = frame.shape[:2]
    scale = TARGET_H / h
    frame = cv2.resize(frame, (int(w * scale), TARGET_H))

    results = model(frame) #Обработка кадра моделью
    result = frame.copy()
    result = put_text_pil(result, f"FPS: {fps:.1f}", (10, 10), font_size=20, color=(0, 200, 255))

    panel = np.full((TARGET_H, PANEL_W, 3), (25, 25, 25), np.uint8) #Создание панели для вывода характеристик
    y = 10
    panel = put_text_pil(panel, "Геометрический анализ", (12, y-10), font_size=25, color=(0, 255, 128))
    y += 30

    objects = [] #Массив объектов
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy() #Получение тензора масок

        for mask in masks:
            gap = False
            mask_uint8 = (mask * 255).astype(np.uint8)
            _, mask_bin = cv2.threshold(mask_uint8, 30, 255, cv2.THRESH_BINARY)

            parts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Нахождение разрывов
            if len(parts) > 1: gap = True 

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2) #Морфологические операции над маской

            fragments, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Поиск отделённых фрагментов
            if not fragments: continue

            merged = np.zeros_like(closed)
            cv2.drawContours(merged, fragments, -1, 255, -1) #Соединенние фрагментов между собой

            final_cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not final_cnts: continue
            cnt = max(final_cnts, key=cv2.contourArea)

            area = cv2.contourArea(cnt) #Нахождение площади
            if area < 50: continue
            perimeter = cv2.arcLength(cnt, True) #Нахождение периметра
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0 #Нахождение центра масс
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            objects.append({"cnt": cnt, "area": area, "perimeter": perimeter, "gap": gap, "cx": cx, "cy": cy}) #Внесение объекта в массив

        objects.sort(key=lambda o: (o["cy"], o["cx"])) #Сортировка объектов по центру масс

    for idx, obj in enumerate(objects):
        area_raw = obj["area"]
        perim_raw = obj["perimeter"]
        compact_raw = (4 * np.pi * area_raw) / (perim_raw ** 2) if perim_raw > 0 else 0
        gap = obj["gap"]

        if idx not in prev_metrics:
            prev_metrics[idx] = {"area": area_raw, "perim": perim_raw, "compact": compact_raw, "gap": gap}
        else:
            #Сглаживание метрик
            prev_metrics[idx]["area"] = ALPHA * area_raw + (1 - ALPHA) * prev_metrics[idx]["area"]
            prev_metrics[idx]["perim"] = ALPHA * perim_raw + (1 - ALPHA) * prev_metrics[idx]["perim"] 
            prev_metrics[idx]["compact"] = ALPHA * compact_raw + (1 - ALPHA) * prev_metrics[idx]["compact"]
            prev_metrics[idx]["gap"] = gap

        area = prev_metrics[idx]["area"]
        perimeter = prev_metrics[idx]["perim"]
        compactness = prev_metrics[idx]["compact"]
        gap = prev_metrics[idx]["gap"]

        color = colors[idx % len(colors)]
        cv2.drawContours(result, [obj["cnt"]], -1, color, 2)
        x, y_rect, _, _ = cv2.boundingRect(obj["cnt"])
        result = put_text_pil(result, f"#{idx+1}", (x, y_rect-10), font_size=16, color=color)

        #Вставка параметров в панель
        if y + 85 < TARGET_H:
            panel = put_text_pil(panel, f"Карандаш #{idx+1}", (15, y), font_size=16, color=color)
            y += 20
            panel = put_text_pil(panel, f"Площадь: {area:.0f} px", (15, y), font_size=14, color=(220,220,220))
            y += 18
            panel = put_text_pil(panel, f"Периметр: {perimeter:.0f} px", (15, y), font_size=14, color=(220,220,220))
            y += 18
            panel = put_text_pil(panel, f"Компактность: {compactness:.2f}", (15, y), font_size=14, color=(220,220,220))
            y += 18
            panel = put_text_pil(panel, f"Перекрытие: {gap}", (15, y), font_size=14, color=(220, 220, 220))
            y += 18
            panel = put_text_pil(panel, f"cx: {cx}, cy: {cy}", (15, y), font_size=14, color=(220, 220, 220))
            y += 25

    output = cv2.hconcat([result, panel]) #Объединие результата обработкия и панели метрик.
    cv2.imshow("Geometry Analyzer", output)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()