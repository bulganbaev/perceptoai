import cv2
import time
import numpy as np
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor

# === 1. ЗАГРУЗКА ПАРАМЕТРОВ КАЛИБРОВКИ ===
calib_data = np.load("data/calibration/calibration_data.npz")

mtxL = calib_data["mtxL"]
distL = calib_data["distL"]
mtxR = calib_data["mtxR"]
distR = calib_data["distR"]
R = calib_data["R"]
T = calib_data["T"]

BASELINE = abs(T[0][0])  # Расстояние между камерами (мм)
FOCAL_LENGTH = mtxL[0, 0]  # Фокусное расстояние в пикселях

print(f"🔧 Загрузка калибровки: baseline={BASELINE:.2f}mm, focal={FOCAL_LENGTH:.2f}px")


def compute_disparity(left_box, right_point):
    """Вычисляет disparity (разницу X) между bbox слева и точкой справа."""
    center_L_x = (left_box[1] + left_box[3]) // 2  # Центр X bbox в левой камере
    center_R_x = right_point[0]  # X-координата найденной точки в правой камере

    disparity = max(1, abs(center_L_x - center_R_x))  # Избегаем деления на 0
    return disparity


def compute_depth(left_boxes, right_points):
    """Вычисление глубины на основе disparity."""
    depths = []
    for left_box, right_pt in zip(left_boxes, right_points):
        disparity = compute_disparity(left_box, right_pt)
        depth = (FOCAL_LENGTH * BASELINE) / disparity  # Глубина в мм
        depths.append((left_box[1], left_box[0], depth))  # (X, Y, Depth)
    return depths


def track_optical_flow(frame_left, frame_right, left_boxes):
    """Оптический поток Lucas-Kanade для поиска точек объектов в правой камере."""
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    left_pts = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (y1, x1, y2, x2) in left_boxes], dtype=np.float32)

    if len(left_pts) == 0:
        return []

    right_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_left, gray_right, left_pts, None)

    matched_points = []
    for i, (new, status_flag) in enumerate(zip(right_pts, status)):
        if status_flag:
            matched_points.append(new)  # Найденная точка в правой камере

    return matched_points


# === 2. ИНИЦИАЛИЗАЦИЯ ДЕТЕКЦИИ ===
inf = HailoInference('data/models/yolov11s.hef', 'data/labels/coco.txt')
proc = Processor(inf, conf=0.5)

# === 3. ЗАПУСК КАМЕР ===
cam_left = CameraDriver(camera_id=0)
cam_right = CameraDriver(camera_id=1)

cam_left.start_camera()
cam_right.start_camera()

print("🎥 Запуск стрима. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            # Запуск детекции только для левой камеры
            detections = proc.process([frame_left])

            # Фильтруем только class=0 (обычно это 'person' в COCO)
            for result in detections:
                filtered_boxes = []
                filtered_scores = []
                filtered_classes = []

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

                frame_left = proc.label_loader.draw_boxes(result)

                # === 4. ОПТИЧЕСКИЙ ПОТОК ДЛЯ ПОИСКА ТОЧЕК В ПРАВОЙ КАМЕРЕ ===
                right_points = track_optical_flow(frame_left, frame_right, filtered_boxes)

                # === 5. ВЫЧИСЛЕНИЕ ГЛУБИНЫ ===
                depth_results = compute_depth(filtered_boxes, right_points)

                # === 6. ОТОБРАЖЕНИЕ ГЛУБИНЫ ===
                for x, y, d in depth_results:
                    cv2.putText(frame_left, f"Depth: {d:.1f} mm", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # Объединяем левый и правый кадр
            combined_frame = cv2.hconcat([frame_left, frame_right])

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
