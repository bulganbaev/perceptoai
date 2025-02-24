import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from cam.camera_driver import StereoCameraSystem
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
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    for i, class_id in enumerate(results['detection_classes']):
        if class_id == 2:
            filtered_boxes.append(results['absolute_boxes'][i])
            filtered_scores.append(results['detection_scores'][i])
            filtered_classes.append(class_id)

    results.update({
        'absolute_boxes': filtered_boxes,
        'detection_classes': filtered_classes,
        'detection_scores': filtered_scores
    })
    return results


def compute_disparity(left_bbox, right_bbox):
    """Вычисляет disparity между bbox в левой и правой камерах"""
    center_L_x = (left_bbox[1] + left_bbox[3]) // 2
    center_R_x = (right_bbox[1] + right_bbox[3]) // 2
    disparity = max(1, abs(center_L_x - center_R_x))  # Избегаем деления на 0
    return disparity, center_L_x, left_bbox[0]  # (disparity, X, Y)


def compute_depth(left_results, right_results, matches, depth_map):
    """Вычисление min/max глубины (1 значение + диапазон)"""
    global depth_history
    depths = {}

    for left_idx, right_idx in matches:
        left_box, right_box = left_results['absolute_boxes'][left_idx], right_results['absolute_boxes'][right_idx]
        disparity, obj_x, obj_y = compute_disparity(left_box, right_box)
        raw_depth = (FOCAL_LENGTH * BASELINE) / disparity  # Глубина в мм

        # Глубина по всей области bbox
        x1, y1, x2, y2 = left_box
        box_depth_values = depth_map[y1:y2, x1:x2]
        box_depth_values = box_depth_values[box_depth_values > 0]

        if len(box_depth_values) > 0:
            min_depth = np.min(box_depth_values)
            max_depth = np.max(box_depth_values)
            filtered_depth = np.median(box_depth_values)
        else:
            min_depth = max_depth = filtered_depth = raw_depth

        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)
        depth_history[obj_id].append(filtered_depth)
        final_depth = np.median(depth_history[obj_id])  # Усредняем depth

        depths[obj_id] = (obj_x, obj_y, final_depth, min_depth, max_depth)  # (X, Y, Depth, Min, Max)

    return list(depths.values())  # Выводим 1 depth + диапазон на объект


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


def draw_boxes(image, results):
    """Отрисовка bbox"""
    for (y1, x1, y2, x2), score in zip(results['absolute_boxes'], results['detection_scores']):
        label = f"Car ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_depth(image, depth_results):
    """Отрисовка глубины (1 значение + диапазон)"""
    for x, y, d, d_min, d_max in depth_results:
        text = f"{d:.1f}mm ({d_min:.1f}-{d_max:.1f}mm)"
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return image


def choose_model():
    """Выбор модели перед запуском"""
    models_dir = "data/models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".hef")]

    print("\n📌 Доступные модели:")
    for i, model in enumerate(model_files):
        print(f"  {i + 1}. {model}")

    while True:
        try:
            choice = int(input("\n👉 Выберите номер модели: ")) - 1
            if 0 <= choice < len(model_files):
                return os.path.join(models_dir, model_files[choice])
            else:
                print("❌ Неверный ввод. Попробуйте снова.")
        except ValueError:
            print("❌ Введите число!")


# === 5. ЗАПУСК КАМЕР И ДЕТЕКЦИИ ===
model_path = choose_model()
inf = HailoInference(model_path)
proc = Processor(inf, conf=0.5)
stereo = StereoCameraSystem()
stereo.start()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")
try:
    while True:
        frame_left, frame_right = stereo.get_synchronized_frames()
        if frame_left is not None and frame_right is not None:
            detections = proc.process([frame_left, frame_right])
            result_left, result_right = filter_people(detections[0]), filter_people(detections[1])
            matches = match_boxes(result_left, result_right)
            depth_map = np.zeros_like(frame_left[:, :, 0])  # Реальную depth_map заменить здесь
            depth_results = compute_depth(result_left, result_right, matches, depth_map)

            processed_left = draw_boxes(frame_left, result_left)
            processed_left = draw_depth(processed_left, depth_results)
            combined = cv2.hconcat([processed_left, frame_right])

            cv2.imshow("Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
