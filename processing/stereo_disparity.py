import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor

# === 1. ЗАГРУЗКА КАЛИБРОВКИ ===
calib_data = np.load("data/calibration/calibration_data.npz")

mtxL, distL = calib_data["mtxL"], calib_data["distL"]
mtxR, distR = calib_data["mtxR"], calib_data["distR"]
R, T = calib_data["R"], calib_data["T"]

BASELINE = abs(T[0][0])  # Расстояние между камерами (мм)
FOCAL_LENGTH = mtxL[0, 0]  # Фокусное расстояние (пиксели)

# === 2. ФИЛЬТР ДЛЯ СТАБИЛЬНОГО DEPTH ===
depth_history = {}
DEPTH_FILTER_SIZE = 5  # Размер скользящего окна

# === 3. ФУНКЦИИ ===
def filter_people(results):
    """Фильтрует только class=0 (человек)"""
    filtered_boxes, filtered_scores = [], []

    for i, class_id in enumerate(results['detection_classes']):
        if class_id == 0:
            filtered_boxes.append(results['absolute_boxes'][i])
            filtered_scores.append(results['detection_scores'][i])

    results.update({'absolute_boxes': filtered_boxes, 'detection_scores': filtered_scores})
    return results


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
    """Сопоставление bbox по X-координатам (Hungarian Algorithm)"""
    left_boxes, right_boxes = left_results["absolute_boxes"], right_results["absolute_boxes"]
    if not left_boxes or not right_boxes:
        return []

    left_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in left_boxes])
    right_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in right_boxes])

    cost_matrix = np.abs(left_centers[:, None] - right_centers[None, :])
    left_indices, right_indices = linear_sum_assignment(cost_matrix)
    return [(l, r) for l, r in zip(left_indices, right_indices)]


def compute_depth(left_results, right_results, matches, depth_map):
    """Вычисление min/max глубины (1 значение + диапазон)"""
    global depth_history
    depths = {}

    for left_idx, right_idx in matches:
        left_box, right_box = left_results['absolute_boxes'][left_idx], right_results['absolute_boxes'][right_idx]
        x1, y1, x2, y2 = left_box

        # Берем среднее по области bbox
        box_depth_values = depth_map[y1:y2, x1:x2]
        box_depth_values = box_depth_values[box_depth_values > 0]

        if len(box_depth_values) > 0:
            min_depth = np.min(box_depth_values)
            max_depth = np.max(box_depth_values)
            filtered_depth = np.median(box_depth_values)
        else:
            min_depth = max_depth = filtered_depth = 0  # Если нет данных

        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)
        depth_history[obj_id].append(filtered_depth)
        final_depth = np.median(depth_history[obj_id])  # Усредняем depth

        depths[obj_id] = (x1, y1, final_depth, min_depth, max_depth)  # (X, Y, Depth, Min, Max)

    return list(depths.values())  # Выводим 1 depth + диапазон на объект


def draw_depth_overlay(image, depth_map, bbox):
    """Накладывает Depth Map на Bounding Box"""
    x1, y1, x2, y2 = bbox
    depth_bbox = depth_map[y1:y2, x1:x2]

    if depth_bbox.size > 0:
        depth_bbox = cv2.normalize(depth_bbox, None, 0, 255, cv2.NORM_MINMAX)
        depth_bbox = cv2.applyColorMap(depth_bbox.astype(np.uint8), cv2.COLORMAP_JET)

        image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 0.5, depth_bbox, 0.5, 0)

    return image


def draw_boxes(image, results, depth_map):
    """Отрисовка bbox + depth overlay"""
    for (y1, x1, y2, x2), score in zip(results['absolute_boxes'], results['detection_scores']):
        image = draw_depth_overlay(image, depth_map, (x1, y1, x2, y2))  # Наложение Depth Map
        label = f"Person ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_depth(image, depth_results):
    """Отрисовка глубины (1 значение + диапазон)"""
    for x, y, d, d_min, d_max in depth_results:
        text = f"{d:.1f}mm ({d_min:.1f}-{d_max:.1f}mm)"
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return image


# === 4. ЗАПУСК КАМЕР И ДЕТЕКЦИИ ===
inf = HailoInference('data/models/yolov11s.hef')
proc = Processor(inf, conf=0.5)

cam_left, cam_right = CameraDriver(camera_id=0), CameraDriver(camera_id=1)
cam_left.start_camera(), cam_right.start_camera()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left, frame_right = cam_left.get_frame(), cam_right.get_frame()
        if frame_left is not None and frame_right is not None:
            depth_map = compute_disparity_map(frame_left, frame_right)  # Карта глубины

            detections = proc.process([frame_left, frame_right])
            result_left, result_right = filter_people(detections[0]), filter_people(detections[1])

            matches = match_boxes(result_left, result_right)
            depth_results = compute_depth(result_left, result_right, matches, depth_map)

            processed_left = draw_boxes(frame_left, result_left, depth_map)  # Теперь с Depth Overlay
            processed_left = draw_depth(processed_left, depth_results)

            combined = cv2.hconcat([processed_left, frame_right])
            cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stereo Depth", 1920, 1080)
            cv2.imshow("Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

cam_left.stop_camera()
cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
