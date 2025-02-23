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
def compute_disparity(left_bbox, right_bbox):
    """Вычисляет disparity между bbox в левой и правой камерах"""
    center_L_x = (left_bbox[1] + left_bbox[3]) // 2
    center_R_x = (right_bbox[1] + right_bbox[3]) // 2
    disparity = max(1, abs(center_L_x - center_R_x))  # Избегаем деления на 0
    return disparity, center_L_x, left_bbox[0]  # (disparity, X, Y)


def compute_depth(left_results, right_results, matches):
    """Вычисление стабильной глубины (1 значение на объект)"""
    global depth_history
    depths = {}

    for left_idx, right_idx in matches:
        left_box, right_box = left_results['absolute_boxes'][left_idx], right_results['absolute_boxes'][right_idx]
        disparity, obj_x, obj_y = compute_disparity(left_box, right_box)
        raw_depth = (FOCAL_LENGTH * BASELINE) / disparity  # Глубина в мм

        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)
        depth_history[obj_id].append(raw_depth)
        final_depth = np.median(depth_history[obj_id])  # Усредняем depth

        depths[obj_id] = (obj_x, obj_y, final_depth)  # (X, Y, Depth)

    return list(depths.values())  # Выводим только 1 depth на объект


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
    for (y1, x1, y2, x2), class_id, score in zip(results['absolute_boxes'], results['detection_classes'], results['detection_scores']):
        if class_id == 0:
            label = f"Person ({score:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_depth(image, depth_results):
    """Отрисовка глубины (1 значение на объект)"""
    for x, y, d in depth_results:
        cv2.putText(image, f"{d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image


# === 4. ВЫБОР МОДЕЛИ ===
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
print(f"🚀 Запуск с моделью: {model_path}")

inf = HailoInference(model_path)
proc = Processor(inf, conf=0.5)

cam_left, cam_right = CameraDriver(camera_id=0), CameraDriver(camera_id=1)
cam_left.start_camera(), cam_right.start_camera()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left, frame_right = cam_left.get_frame(), cam_right.get_frame()
        if frame_left is not None and frame_right is not None:
            detections = proc.process([frame_left, frame_right])
            result_left, result_right = detections[0], detections[1]

            matches = match_boxes(result_left, result_right)
            depth_results = compute_depth(result_left, result_right, matches)

            processed_left = draw_boxes(frame_left, result_left)
            processed_left = draw_depth(processed_left, depth_results)  # 1 depth на объект

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
