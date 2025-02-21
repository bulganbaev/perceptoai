import cv2
import time
import numpy as np
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor


def draw_boxes(image, results: dict):
    """Отрисовка боксов на изображении."""
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

# Запуск двух камер
cam_left = CameraDriver(camera_id=0)
cam_right = CameraDriver(camera_id=1)

cam_left.start_camera()
cam_right.start_camera()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            # Запуск детекции на обеих камерах
            detections = proc.process([frame_left, frame_right])

            # Обработка левой камеры
            result_left = detections[0]
            filtered_boxes_left = []
            filtered_scores_left = []
            filtered_classes_left = []

            for i, class_id in enumerate(result_left['detection_classes']):
                if class_id == 0:  # Фильтруем только class=0
                    filtered_boxes_left.append(result_left['absolute_boxes'][i])
                    filtered_scores_left.append(result_left['detection_scores'][i])
                    filtered_classes_left.append(class_id)

            result_left.update({
                'absolute_boxes': filtered_boxes_left,
                'detection_classes': filtered_classes_left,
                'detection_scores': filtered_scores_left
            })

            processed_left = draw_boxes(frame_left, result_left)

            # Обработка правой камеры
            result_right = detections[1]
            filtered_boxes_right = []
            filtered_scores_right = []
            filtered_classes_right = []

            for i, class_id in enumerate(result_right['detection_classes']):
                if class_id == 0:  # Фильтруем только class=0
                    filtered_boxes_right.append(result_right['absolute_boxes'][i])
                    filtered_scores_right.append(result_right['detection_scores'][i])
                    filtered_classes_right.append(class_id)

            result_right.update({
                'absolute_boxes': filtered_boxes_right,
                'detection_classes': filtered_classes_right,
                'detection_scores': filtered_scores_right
            })

            processed_right = draw_boxes(frame_right, result_right)

            combined = cv2.hconcat([processed_left, processed_right])
            cv2.namedWindow("Dual Cameras", cv2.WINDOW_NORMAL)  # Делаем окно изменяемым
            cv2.resizeWindow("Dual Cameras", 1920, 1080)
            cv2.imshow("Dual Cameras", combined)

        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# Завершаем работу камер
cam_left.stop_camera()
cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
