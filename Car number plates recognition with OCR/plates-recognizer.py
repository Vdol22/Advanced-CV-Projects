import numpy as np
from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, write_csv, license_complies_format, format_license, read_license_plate

model = YOLO("yolov8n.yaml") 

# Обучение
model = model.train(data='data.yaml', epochs=30, device=0, workers=2, imgsz=256) 

# Загрузка видео
cap = cv2.VideoCapture('./sample.mp4')

# Загрузим модели для определния машин и номеров
coco_model = YOLO('yolov8n.pt') # Для машин
licence_detector = YOLO('./runs/detect/train3/weights/last.pt')
auto_tracker = Sort() # Трекер из библиотеки Sort

# Переменные для ввода/вывода
vehicles = [2, 3, 5, 7] # Список ID транспортных средств, чтобы модель не определяла всё, чему обучена
frame_num = -1
results = {}

# Чтение видео
ret = True
while ret:
    frame_num += 1
    ret, frame = cap.read()
    if ret:
        results[frame_num] = {}
        # Определяем авто
        auto_detections = coco_model(frame)[0]
        detections_ = [] # Хранит определния всех авто
        for detection in auto_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Отслеживание машин
        track_ids = auto_tracker.update(np.asarray(detections_)) # Хранит bboxs всех определённых машин

        # Определение номеров
        plates_detections = licence_detector(frame)[0]
        for plate in plates_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate

            # Привязка номера к конкретной машине
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)

            if car_id != -1:
                # Выделение номеров
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Распознование символов на номерах
                # Бинаризация номеров
                gray_plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, thresh_plate_crop = cv2.threshold(gray_plate_crop, 64, 255, cv2.THRESH_BINARY_INV) # Ниже 64 все пиксели максятся до 255, ниже до 0
                plate_text, plate_text_conf_score = read_license_plate(thresh_plate_crop)

                if plate_text is not None:
                    results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}, 
                                                'license_plate': {'bbox': [x1, y1, x2, y2], 
                                                                'text': plate_text, 
                                                                "bbox_score": score, 
                                                                'text_score': plate_text_conf_score}}


# Сохренение вывода текста
write_csv(results, './test.csv')