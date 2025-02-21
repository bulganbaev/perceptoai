import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm
from cam.camera_driver import CameraDriver
from processing.hailo_detection import HailoInference, Processor

# === 1. ЗАГРУЗКА КАЛИБРОВКИ ===
calib_data = np.load("data/calibration/calibration_data.npz")
mtxL, distL, mtxR, distR, R, T = calib_data["mtxL"], calib_data["distL"], calib_data["mtxR"], calib_data["distR"], calib_data["R"], calib_data["T"]

BASELINE = abs(T[0][0])  # Расстояние между камерами (мм)
FOCAL_LENGTH = mtxL[0, 0]  # Фокусное расстояние в пикселях
print(f"🔧 Калибровка загружена: baseline={BASELINE:.2f}mm, focal={FOCAL_LENGTH:.2f}px")


def choose_model():
    """Выбираем модель перед запуском."""
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
            print("❌ Неверный ввод. Попробуйте снова.")
        except ValueError:
            print("❌ Введите число!")


def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений на изображении."""
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def compute_disparity(left_box, right_box):
    """Вычисление disparity между левым и правым bbox."""
    center_L_x = (left_box[1] + left_box[3]) // 2
    center_R_x = (right_box[1] + right_box[3]) // 2

    disparity = max(1, abs(center_L_x - center_R_x))  # Избегаем деления на 0
    return disparity, center_L_x, (left_box[0] + left_box[2]) // 2


def compute_depth(left_results, right_results, matches):
    """Вычисление глубины с учетом disparity."""
    return [(compute_disparity(left_results['absolute_boxes'][i], right_results['absolute_boxes'][j])[-2:],
             (FOCAL_LENGTH * BASELINE) / compute_disparity(left_results['absolute_boxes'][i], right_results['absolute_boxes'][j])[0])
            for i, j in matches]


def compute_iou(boxA, boxB):
    """Вычисление IoU между двумя bbox."""
    xA, yA, xB, yB = max(boxA[1], boxB[1]), max(boxA[0], boxB[0]), min(boxA[3], boxB[3]), min(boxA[2], boxB[2])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area, boxB_area = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0]), (boxB[3] - boxB[1]) * (boxB[2] - boxB[0])
    return inter_area / float(boxA_area + boxB_area - inter_area) if boxA_area + boxB_area - inter_area > 0 else 0


def match_boxes(left_results, right_results, iou_threshold=0.9):
    """Сопоставление bbox с помощью Hungarian Algorithm."""
    left_boxes, right_boxes = left_results['absolute_boxes'], right_results['absolute_boxes']
    if not left_boxes or not right_boxes:
        return []

    cost_matrix = np.array([[1 - compute_iou(l, r) for r in right_boxes] for l in left_boxes])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < (1 - iou_threshold)]


def draw_boxes(image, results, color=(0, 255, 0)):
    """Отрисовка bbox."""
    for (y1, x1, y2, x2), class_id, score in zip(results['absolute_boxes'], results['detection_classes'], results['detection_scores']):
        if class_id == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Person ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image


def draw_depth(image, depth_results):
    """Отрисовка глубины на изображении."""
    for (x, y), d in depth_results:
        cv2.putText(image, f"Depth: {d:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return image


def process_frame(frame, mtx, dist, processor):
    """Обработка кадра: коррекция, детекция и рисование bbox."""
    frame = undistort_and_rectify(frame, mtx, dist)
    result = processor.process([frame])[0]
    return draw_boxes(frame, result), result


# === 2. ВЫБОР МОДЕЛИ ===
model_path = choose_model()
print(f"🚀 Запуск с моделью: {model_path}")

inf, proc = HailoInference(model_path), Processor(HailoInference(model_path), conf=0.5)
cam_left, cam_right = CameraDriver(camera_id=0), CameraDriver(camera_id=1)
cam_left.start_camera(), cam_right.start_camera()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left, frame_right = cam_left.get_frame(), cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            processed_left, result_left = process_frame(frame_left, mtxL, distL, proc)
            processed_right, result_right = process_frame(frame_right, mtxR, distR, proc)

            matches = match_boxes(result_left, result_right)
            print(f"🔍 Matches: {matches}")

            depth_results = compute_depth(result_left, result_right, matches)

            processed_left, processed_right = draw_depth(processed_left, depth_results), draw_depth(processed_right, depth_results)

            combined = cv2.hconcat([processed_left, processed_right])
            cv2.imshow("Stereo Depth", cv2.resize(combined, (1920, 1080)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка...")

cam_left.stop_camera(), cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
