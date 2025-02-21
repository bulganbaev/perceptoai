import cv2
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


def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений на изображении."""
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def draw_boxes(image, results, color=(0, 255, 0)):
    """Отрисовка баундинг боксов."""
    for (y1, x1, y2, x2), class_id, score in zip(results['absolute_boxes'], results['detection_classes'],
                                                 results['detection_scores']):
        if class_id == 0:  # Только class=0
            label = f"{class_id} ({score:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def match_boxes(left_results, right_results):
    """Сопоставление объектов по высоте и X-координатам."""
    matches = []
    left_boxes, right_boxes = left_results['absolute_boxes'], right_results['absolute_boxes']

    for i, (y1_L, x1_L, y2_L, x2_L) in enumerate(left_boxes):
        center_L = ((x1_L + x2_L) // 2, (y1_L + y2_L) // 2)
        best_match, best_distance = None, float('inf')

        for j, (y1_R, x1_R, y2_R, x2_R) in enumerate(right_boxes):
            center_R = ((x1_R + x2_R) // 2, (y1_R + y2_R) // 2)
            y_diff = abs(center_L[1] - center_R[1])  # Разница по Y
            x_diff = abs(center_L[0] - center_R[0])  # Разница по X

            if y_diff < 20 and x_diff < best_distance:
                best_distance, best_match = x_diff, j

        if best_match is not None:
            matches.append((i, best_match))

    return matches


def compute_depth(left_results, right_results, matches):
    """Вычисление глубины по диспарити с учетом калибровки."""
    depths = []
    for i, j in matches:
        center_L = (left_results['absolute_boxes'][i][1] + left_results['absolute_boxes'][i][3]) // 2
        center_R = (right_results['absolute_boxes'][j][1] + right_results['absolute_boxes'][j][3]) // 2

        disparity = abs(center_L - center_R)
        depth = (FOCAL_LENGTH * BASELINE) / disparity if disparity > 0 else float('inf')

        depths.append((center_L, left_results['absolute_boxes'][i][0], depth))

    return depths


# === 2. ЗАПУСК КАМЕР И ДЕТЕКЦИИ ===
inf = HailoInference('data/models/yolov11s.hef')
proc = Processor(inf, conf=0.5)

cam_left = CameraDriver(camera_id=0)
cam_right = CameraDriver(camera_id=1)
cam_left.start_camera()
cam_right.start_camera()

print("🎥 Запуск стереопотока с расчетом глубины. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            # === 3. Исправляем искажения ===
            frame_left = undistort_and_rectify(frame_left, mtxL, distL)
            frame_right = undistort_and_rectify(frame_right, mtxR, distR)

            # === 4. Запускаем детекцию ===
            detections = proc.process([frame_left, frame_right])
            result_left, result_right = detections[0], detections[1]

            # === 5. Сопоставляем объекты ===
            matches = match_boxes(result_left, result_right)

            # === 6. Вычисляем глубину ===
            depth_results = compute_depth(result_left, result_right, matches)

            # === 7. Отрисовка боксов и глубины ===
            processed_left = draw_boxes(frame_left, result_left, color=(0, 255, 0))
            processed_right = draw_boxes(frame_right, result_right, color=(255, 0, 0))

            for x, y, d in depth_results:
                cv2.putText(processed_left, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 2)
                cv2.putText(processed_right, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 2)

            # === 8. Вывод ===
            combined = cv2.hconcat([processed_left, processed_right])
            cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stereo Depth", 1920, 1080)
            cv2.imshow("Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# === 9. Завершаем работу камер ===
cam_left.stop_camera()
cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
