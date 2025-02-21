import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
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


def compute_disparity(left_box, right_box):
    """Вычисляет disparity между левым и правым bbox."""
    center_L = (left_box[1] + left_box[3]) // 2  # X-координата центра левого bbox
    center_R = (right_box[1] + right_box[3]) // 2  # X-координата центра правого bbox

    disparity = max(1, abs(center_L - center_R))  # Избегаем деления на 0
    return disparity


def compute_depth(left_results, right_results, matches):
    """Вычисление глубины с учетом disparity."""
    depths = []
    for i, j in matches:
        left_box = left_results['absolute_boxes'][i]
        right_box = right_results['absolute_boxes'][j]

        disparity = compute_disparity(left_box, right_box)
        depth = (FOCAL_LENGTH * BASELINE) / disparity  # Глубина в мм

        depths.append((left_box[1], left_box[0], depth))  # (X, Y, Depth)

    return depths


def compute_iou(boxA, boxB):
    """Вычисляет IoU между двумя bbox."""
    (y1_A, x1_A, y2_A, x2_A) = boxA
    (y1_B, x1_B, y2_B, x2_B) = boxB

    x_left = max(x1_A, x1_B)
    y_top = max(y1_A, y1_B)
    x_right = min(x2_A, x2_B)
    y_bottom = min(y2_A, y2_B)

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    boxA_area = (x2_A - x1_A) * (y2_A - y1_A)
    boxB_area = (x2_B - x1_B) * (y2_B - y1_B)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area) if (boxA_area + boxB_area - intersection_area) > 0 else 0
    return iou


def match_boxes(left_results, right_results, iou_threshold=0.5):
    """Сопоставление bbox с помощью Hungarian Algorithm."""
    left_boxes = left_results['absolute_boxes']
    right_boxes = right_results['absolute_boxes']

    if len(left_boxes) == 0 or len(right_boxes) == 0:
        return []  # Если нет объектов, просто возвращаем пустой список

    cost_matrix = np.zeros((len(left_boxes), len(right_boxes)))

    for i, left_box in enumerate(left_boxes):
        for j, right_box in enumerate(right_boxes):
            iou = compute_iou(left_box, right_box)
            cost_matrix[i, j] = 1 - iou  # Чем ниже, тем лучше соответствие

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < (1 - iou_threshold):
            matches.append((i, j))

    return matches


def draw_boxes(image, results, color=(0, 255, 0)):
    """Отрисовка bbox."""
    for (y1, x1, y2, x2), class_id, score in zip(results['absolute_boxes'], results['detection_classes'],
                                                 results['detection_scores']):
        if class_id == 0:
            label = f"Person ({score:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image


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

            # === 5. Сопоставляем bbox (Hungarian Algorithm) ===
            matches = match_boxes(result_left, result_right)

            # === 6. Вычисляем глубину ===
            depth_results = compute_depth(result_left, result_right, matches)

            # === 7. Отрисовка bbox и глубины ===
            processed_left = draw_boxes(frame_left, result_left, color=(0, 255, 0))
            processed_right = draw_boxes(frame_right, result_right, color=(255, 0, 0))

            for x, y, d in depth_results:
                cv2.putText(processed_left, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(processed_right, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

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
