import cv2
import time
import numpy as np
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor


def draw_boxes(image, results: dict):
    boxes = results.get('absolute_boxes', [])
    scores = results.get('detection_scores', [])
    classes = results.get('detection_classes', [])

    for i, (y1, x1, y2, x2) in enumerate(boxes):
        class_id = classes[i] if i < len(classes) else "Unknown"
        score = scores[i] if i < len(scores) else 0.0
        label = f'{class_id} ({score:.2f})'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Инициализация модели детекции
inf = HailoInference('data/models/yolov11s.hef')
proc = Processor(inf, conf=0.5)

# Запуск одной камеры (например, левой)
cam = CameraDriver(camera_id=0)
cam.start_camera()

print("🎥 Запуск потока. Нажмите 'q' для выхода.")

try:
    while True:
        frame = cam.get_frame()

        if frame is not None:
            # Запуск детекции
            detections = proc.process([frame])
            print(detections)

            # Фильтруем только class=0 (обычно это 'person' в COCO)
            filtered_boxes = []
            filtered_scores = []
            filtered_classes = []

            result = detections[0]  # Обрабатываем первый кадр (только одна камера)

            for i, class_id in enumerate(result['detection_classes']):
                if class_id == 0:  # Фильтруем только class=0
                    filtered_boxes.append(result['absolute_boxes'][i])
                    filtered_scores.append(result['detection_scores'][i])
                    filtered_classes.append(class_id)

            # Обновляем результаты и рисуем боксы только для class=0
            result.update({
                'absolute_boxes': filtered_boxes,
                'detection_classes': filtered_classes,
                'detection_scores': filtered_scores
            })

            processed_frame = draw_boxes(frame, result)

            # Отображение окна
            cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera Stream", 1280, 720)
            cv2.imshow("Camera Stream", processed_frame)

        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# Завершаем работу камеры
cam.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
