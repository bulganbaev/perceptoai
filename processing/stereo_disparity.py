import cv2
import os
import numpy as np
from collections import deque
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor

# === 1. ЗАГРУЗКА ПАРАМЕТРОВ КАЛИБРОВКИ ===
calib_data = np.load("data/calibration/calibration_data.npz")

mtxL, distL = calib_data["mtxL"], calib_data["distL"]
mtxR, distR = calib_data["mtxR"], calib_data["distR"]
R, T = calib_data["R"], calib_data["T"]

BASELINE = abs(T[0][0])  # Расстояние между камерами (мм)
FOCAL_LENGTH = mtxL[0, 0]  # Фокусное расстояние в пикселях

print(f"🔧 Калибровка загружена: baseline={BASELINE:.2f}mm, focal={FOCAL_LENGTH:.2f}px")

models_dir = "data/models"
depth_history = {}
DEPTH_FILTER_SIZE = 5  # Размер медианного фильтра


def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений."""
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def track_features_lk(prev_frame, next_frame, prev_points):
    """Оптический поток Lucas-Kanade для трекинга точек."""
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_points, None, **lk_params)

    valid_points = next_points[status == 1] if status is not None else []
    return valid_points


def compute_disparity_lk(left_frame, right_frame, left_boxes):
    """Определение диспарити с помощью LK Optical Flow."""
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    depths = []
    for i, (y1, x1, y2, x2) in enumerate(left_boxes):
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        feature_points = np.array([[center_x, center_y]], dtype=np.float32)

        tracked_points = track_features_lk(gray_left, gray_right, feature_points)

        if len(tracked_points) > 0:
            right_x = int(tracked_points[0][0])
            disparity = max(1, abs(center_x - right_x))
            raw_depth = (FOCAL_LENGTH * BASELINE) / disparity

            if i not in depth_history:
                depth_history[i] = deque(maxlen=DEPTH_FILTER_SIZE)

            depth_history[i].append(raw_depth)
            final_depth = np.median(depth_history[i])
            depths.append((center_x, center_y, final_depth))

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
    """Отрисовка глубины на левом кадре."""
    for x, y, d in depth_results:
        cv2.putText(image, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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
            print("❌ Неверный ввод. Попробуйте снова.")
        except ValueError:
            print("❌ Введите число!")


# === 3. ЗАПУСК КАМЕР И МОДЕЛИ ===
model_path = choose_model()
print(f"🚀 Запуск с моделью: {model_path}")

inf = HailoInference(model_path)
proc = Processor(inf, conf=0.5)

cam_left = CameraDriver(camera_id=0)
cam_right = CameraDriver(camera_id=1)
cam_left.start_camera()
cam_right.start_camera()

print("🎥 Запуск стереопотока с оптическим потоком Lucas-Kanade. Нажмите 'q' для выхода.")

try:
    prev_left_frame = None

    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            frame_left = undistort_and_rectify(frame_left, mtxL, distL)
            frame_right = undistort_and_rectify(frame_right, mtxR, distR)

            detections = proc.process([frame_left])
            result_left = detections[0]

            filtered_boxes = []
            for i, class_id in enumerate(result_left['detection_classes']):
                if class_id == 0:
                    filtered_boxes.append(result_left['absolute_boxes'][i])

            result_left.update({'absolute_boxes': filtered_boxes})

            if prev_left_frame is not None:
                depth_results = compute_disparity_lk(prev_left_frame, frame_right, filtered_boxes)
            else:
                depth_results = []

            processed_left = draw_boxes(frame_left, result_left)
            processed_left = draw_depth(processed_left, depth_results)

            combined = cv2.hconcat([processed_left, frame_right])
            cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stereo Depth", 1920, 1080)
            cv2.imshow("Stereo Depth", combined)

            prev_left_frame = frame_left.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

cam_left.stop_camera()
cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
