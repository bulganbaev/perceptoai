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

# === 2. ИНИЦИАЛИЗАЦИЯ ОЧЕРЕДИ ДЛЯ ФИЛЬТРАЦИИ ГЛУБИНЫ ===
depth_history = {}  # Храним последние измерения глубины
DEPTH_FILTER_SIZE = 5  # Количество последних измерений для медианного фильтра


def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений на изображении."""
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def track_optical_flow(frame_left, frame_right, left_boxes):
    """Оптический поток для сопоставления объектов."""
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    left_pts = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for (y1, x1, y2, x2) in left_boxes], dtype=np.float32)

    if len(left_pts) == 0:
        return []

    right_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_left, gray_right, left_pts, None)

    matches = []
    for i, (new, status_flag) in enumerate(zip(right_pts, status)):
        if status_flag:
            matches.append((i, new))  # (индекс в левой камере, найденная точка в правой камере)

    return matches


def compute_depth(left_results, right_results, matches):
    """Вычисление глубины с учетом disparity и медианного фильтра."""
    global depth_history
    depths = []
    left_boxes = left_results['absolute_boxes']
    right_boxes = right_results['absolute_boxes']

    for i, right_pt in matches:
        left_box = left_boxes[i]

        best_match, min_dist = None, float('inf')
        for j, right_box in enumerate(right_boxes):
            center_R_x = (right_box[1] + right_box[3]) // 2  # Центр X bbox справа
            dist = abs(center_R_x - right_pt[0])

            if dist < min_dist:
                min_dist, best_match = dist, j

        if best_match is not None:
            right_box = right_boxes[best_match]
            disparity = max(1, abs(left_box[1] - right_box[1]))  # Избегаем деления на 0
            depth = (FOCAL_LENGTH * BASELINE) / disparity

            # 📌 Фильтрация глубины через медианный фильтр
            obj_id = (left_box[1], left_box[0])  # Идентификатор объекта
            if obj_id not in depth_history:
                depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)

            depth_history[obj_id].append(depth)
            filtered_depth = np.median(depth_history[obj_id])  # Медианное значение

            depths.append((left_box[1], left_box[0], filtered_depth))  # (X, Y, Depth)

    return depths


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
    """Выбираем модель перед запуском"""
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

            matches = track_optical_flow(frame_left, frame_right, result_left['absolute_boxes'])
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
