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
models_dir = "data/models"

# === 2. ИНИЦИАЛИЗАЦИЯ ФИЛЬТРА ГЛУБИНЫ ===
depth_history = {}  # Кэш глубины для медианного фильтра
DEPTH_FILTER_SIZE = 5  # Количество последних измерений для медианного фильтра


def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений на изображении."""
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def match_boxes(left_results, right_results):
    """Сопоставление объектов между левым и правым изображением."""
    left_boxes = left_results["absolute_boxes"]
    right_boxes = right_results["absolute_boxes"]

    if not left_boxes or not right_boxes:
        return []

    # Вычисляем центры bounding box'ов
    left_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in left_boxes])
    right_centers = np.array([(x1 + x2) // 2 for (_, x1, _, x2) in right_boxes])

    # Создаём матрицу расстояний
    cost_matrix = np.abs(left_centers[:, None] - right_centers[None, :])

    # Применяем алгоритм Венгера для нахождения оптимального соответствия
    left_indices, right_indices = linear_sum_assignment(cost_matrix)

    # Формируем список сопоставлений (индекс_в_левом, индекс_в_правом)
    matches = [(l, r) for l, r in zip(left_indices, right_indices)]

    return matches


def compute_depth(left_results, right_results, matches):
    """Вычисление глубины для каждого bounding box'а."""
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
        depth = (FOCAL_LENGTH * BASELINE) / disparity

        # 📌 Фильтрация через медианный фильтр
        obj_id = left_idx  # Один box = один depth
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)

        depth_history[obj_id].append(depth)
        filtered_depth = np.median(depth_history[obj_id])  # Медианное значение

        # ✅ Теперь каждый bounding box получает только одно значение глубины
        depths[obj_id] = (left_box[1], left_box[0], filtered_depth)

    return list(depths.values())  # Возвращаем список значений



def draw_boxes(image, results, color=(0, 255, 0)):
    """Отрисовка bbox."""
    for (y1, x1, y2, x2), class_id, score in zip(results['absolute_boxes'], results['detection_classes'],
                                                 results['detection_scores']):
        if class_id == 0:
            label = f"Person ({score:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image


def draw_depth(image, depth_results):
    """Отрисовка метки глубины на изображении."""
    for x, y, d in depth_results:
        cv2.putText(image, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image


def choose_model():
    """Выбор модели перед запуском."""
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


# === 3. ЗАПУСК КАМЕР И ДЕТЕКЦИИ ===
model_path = choose_model()
print(f"🚀 Запуск с моделью: {model_path}")

inf = HailoInference(model_path)
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

            detections = proc.process([frame_left, frame_right])
            result_left, result_right = detections[0], detections[1]

            matches = match_boxes(result_left, result_right)
            depth_results = compute_depth(result_left, result_right, matches)

            processed_left = draw_boxes(frame_left, result_left, color=(0, 255, 0))
            processed_right = draw_boxes(frame_right, result_right, color=(255, 0, 0))

            processed_left = draw_depth(processed_left, depth_results)
            processed_right = draw_depth(processed_right, depth_results)

            combined = cv2.hconcat([processed_left, processed_right])
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
