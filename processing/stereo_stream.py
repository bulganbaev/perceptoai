import cv2
import time
import numpy as np
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor

# Инициализация модели детекции для обеих камер
inf = HailoInference('data/models/yolov11s.hef', 'data/labels/coco.txt')
proc = Processor(inf, conf=0.5)

# Запуск стереокамер
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
            # Запуск детекции на обоих кадрах
            detections = proc.process([frame_left, frame_right])

            filtered_frames = []
            for i, frame in enumerate([frame_left, frame_right]):
                result = detections[i]

                # Фильтруем только class=0 (обычно это 'person' в COCO)
                filtered_boxes = []
                filtered_scores = []
                filtered_classes = []

                for j, class_id in enumerate(result['detection_classes']):
                    if class_id == 0:  # Фильтруем только class=0
                        filtered_boxes.append(result['absolute_boxes'][j])
                        filtered_scores.append(result['detection_scores'][j])
                        filtered_classes.append(class_id)

                # Обновляем результаты и рисуем боксы только для class=0
                result.update({
                    'absolute_boxes': filtered_boxes,
                    'detection_classes': filtered_classes,
                    'detection_scores': filtered_scores
                })

                filtered_frames.append(proc.label_loader.draw_boxes(result))

            # Объединяем обработанные кадры
            combined_frame = cv2.hconcat(filtered_frames)

            # Отображение окна
            cv2.namedWindow("Stereo Stream", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stereo Stream", 1920, 1080)
            cv2.imshow("Stereo Stream", combined_frame)

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
