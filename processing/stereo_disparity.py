import cv2
import os
import numpy as np
from collections import deque
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

# === 2. ИНИЦИАЛИЗАЦИЯ ФИЛЬТРА ГЛУБИНЫ ===
depth_history = {}
DEPTH_FILTER_SIZE = 5  # Размер фильтра


def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений на изображении."""
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def compute_disparity_map(left_img, right_img):
    """Рассчет карты глубины через disparity."""
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)

    disparity[disparity <= 0] = 1  # Избегаем деления на 0
    depth_map = (FOCAL_LENGTH * BASELINE) / disparity  # Глубина по disparity

    return depth_map


def match_boxes(left_results, right_results):
    """Сопоставление bounding box'ов по центрам и горизонтали."""
    left_boxes = left_results["absolute_boxes"]
    right_boxes = right_results["absolute_boxes"]

    if not left_boxes or not right_boxes:
        return []

    left_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in left_boxes])
    right_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in right_boxes])

    cost_matrix = np.abs(left_centers[:, None] - right_centers[None, :])

    left_indices, right_indices = linear_sum_assignment(cost_matrix)
    matches = [(l, r) for l, r in zip(left_indices, right_indices)]

    return matches


def compute_depth(left_results, right_results, matches, depth_map):
    """Вычисление глубины на основе disparity + медианный фильтр depth map."""
    global depth_history
    depths = {}
    left_boxes = left_results['absolute_boxes']
    right_boxes = right_results['absolute_boxes']

    for left_idx, right_idx in matches:
        left_box = left_boxes[left_idx]
        right_box = right_boxes[right_idx]

        left_center_x = (left_box[1] + left_box[3]) // 2
        right_center_x = (right_box[1] + right_box[3]) // 2

        disparity = max(1, abs(left_center_x - right_center_x))  # Избегаем деления на 0
        raw_depth = (FOCAL_LENGTH * BASELINE) / disparity

        x1, y1, x2, y2 = left_box
        box_depth_values = depth_map[y1:y2, x1:x2]
        box_depth_values = box_depth_values[box_depth_values > 0]

        if len(box_depth_values) > 0:
            filtered_depth = np.median(box_depth_values)
        else:
            filtered_depth = raw_depth

        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)

        depth_history[obj_id].append(filtered_depth)
        final_depth = np.median(depth_history[obj_id])

        depths[obj_id] = (left_box[1], left_box[0], final_depth)

    return list(depths.values())


def filter_objects(results):
    """Оставляет только объекты с class_id == 0 (человек)."""
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    for i, class_id in enumerate(results['detection_classes']):
        if class_id == 0:  # Только человек
            filtered_boxes.append(results['absolute_boxes'][i])
            filtered_scores.append(results['detection_scores'][i])
            filtered_classes.append(class_id)

    results.update({
        'absolute_boxes': filtered_boxes,
        'detection_classes': filtered_classes,
        'detection_scores': filtered_scores
    })


def draw_boxes(image, results, color=(0, 255, 0)):
    """Отрисовка bbox."""
    for (y1, x1, y2, x2), class_id, score in zip(results['absolute_boxes'], results['detection_classes'],
                                                 results['detection_scores']):
        label = f"Person ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image


def draw_depth(image, depth_results):
    """Отрисовка глубины на изображении."""
    for x, y, d in depth_results:
        cv2.putText(image, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image


# === 3. ЗАПУСК КАМЕР И МОДЕЛИ ===
inf = HailoInference("data/models/yolov11s.hef")
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
            frame_left = undistort_and_rectify(frame_left, mtxL, distL)
            frame_right = undistort_and_rectify(frame_right, mtxR, distR)

            depth_map = compute_disparity_map(frame_left, frame_right)  # Расчет карты глубины

            detections = proc.process([frame_left, frame_right])
            result_left, result_right = detections[0], detections[1]

            filter_objects(result_left)
            filter_objects(result_right)

            matches = match_boxes(result_left, result_right)
            depth_results = compute_depth(result_left, result_right, matches, depth_map)

            processed_left = draw_boxes(frame_left, result_left)
            processed_right = draw_boxes(frame_right, result_right)

            processed_left = draw_depth(processed_left, depth_results)

            combined = cv2.hconcat([processed_left, processed_right])

            cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)  # Изменяемое окно
            cv2.resizeWindow("Stereo Depth", 1920, 1080)  # Устанавливаем размер окна
            cv2.imshow("Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

cam_left.stop_camera()
cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
