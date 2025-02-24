import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
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


# === 3. УЛУЧШЕННЫЕ ФУНКЦИИ ===
def filter_people(results):
    """Фильтрует только class=0 (человек)"""
    indices = [i for i, cls in enumerate(results['detection_classes']) if cls == 2]
    return {key: [results[key][i] for i in indices] for key in results}


def compute_disparity(left_bbox, right_bbox, depth_map):
    """Улучшенный расчет disparity"""
    x1, y1, x2, y2 = left_bbox
    x1_r, y1_r, x2_r, y2_r = right_bbox

    left_region = depth_map[y1:y2, x1:x2]
    right_region = depth_map[y1_r:y2_r, x1_r:x2_r]

    if left_region.size > 0 and right_region.size > 0:
        left_x = np.mean(np.where(left_region > 0)[1])
        right_x = np.mean(np.where(right_region > 0)[1])
        disparity = max(1, abs(left_x - right_x))
    else:
        disparity = max(1, abs((x1 + x2) // 2 - (x1_r + x2_r) // 2))

    return disparity, (x1 + x2) // 2, y1


def compute_depth(left_results, right_results, matches, depth_map):
    """Вычисление глубины с улучшенной фильтрацией"""
    global depth_history
    depths = {}

    for left_idx, right_idx in matches:
        left_box, right_box = left_results['absolute_boxes'][left_idx], right_results['absolute_boxes'][right_idx]
        disparity, obj_x, obj_y = compute_disparity(left_box, right_box, depth_map)
        raw_depth = (FOCAL_LENGTH * BASELINE) / disparity

        # Фильтрация глубины
        x1, y1, x2, y2 = left_box
        box_depth_values = depth_map[y1:y2, x1:x2].flatten()
        box_depth_values = box_depth_values[box_depth_values > 0]

        if len(box_depth_values) > 0:
            clustering = DBSCAN(eps=5, min_samples=3).fit(box_depth_values.reshape(-1, 1))
            cluster_labels = clustering.labels_
            largest_cluster = max(set(cluster_labels), key=list(cluster_labels).count)
            filtered_depth = np.median(box_depth_values[cluster_labels == largest_cluster])
        else:
            filtered_depth = raw_depth

        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)
        depth_history[obj_id].append(filtered_depth)
        final_depth = np.median(depth_history[obj_id])

        depths[obj_id] = (obj_x, obj_y, final_depth)

    return list(depths.values())


def match_boxes(left_results, right_results):
    """Сопоставление bbox с учетом размеров"""
    left_boxes, right_boxes = left_results["absolute_boxes"], right_results["absolute_boxes"]
    if not left_boxes or not right_boxes:
        return []

    left_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in left_boxes])
    right_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in right_boxes])
    size_diffs = np.array(
        [abs((x2 - x1) - (rx2 - rx1)) for (x1, _, x2, _), (rx1, _, rx2, _) in zip(left_boxes, right_boxes)])

    cost_matrix = np.abs(left_centers[:, None] - right_centers[None, :]) + size_diffs[:, None]
    left_indices, right_indices = linear_sum_assignment(cost_matrix)
    return [(l, r) for l, r in zip(left_indices, right_indices)]


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


# === 5. ЗАПУСК ===
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
