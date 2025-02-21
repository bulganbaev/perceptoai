import cv2
import os
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
models_dir = "data/models"



def undistort_and_rectify(frame, mtx, dist):
    """Исправление искажений на изображении."""
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def compute_disparity(left_box, right_box):
    """Вычисляет disparity между левым и правым bbox."""
    center_L_x = (left_box[1] + left_box[3]) // 2  # X-координата центра bbox слева
    center_R_x = (right_box[1] + right_box[3]) // 2  # X-координата центра bbox справа

    height_L = left_box[2] - left_box[0]  # Высота объекта слева
    height_R = right_box[2] - right_box[0]  # Высота объекта справа

    # Если bbox очень высокий → берём только середину
    if height_L > 100 or height_R > 100:
        center_L_y = left_box[0] + height_L // 3  # 1/3 сверху
        center_R_y = right_box[0] + height_R // 3  # 1/3 сверху
    else:
        center_L_y = (left_box[0] + left_box[2]) // 2  # Обычный центр
        center_R_y = (right_box[0] + right_box[2]) // 2  # Обычный центр

    disparity = max(1, abs(center_L_x - center_R_x))  # Избегаем деления на 0
    return disparity, center_L_x, center_L_y


def compute_depth(left_results, right_results, matches):
    """Вычисление глубины с учетом disparity и реального объекта."""
    depths = []
    for i, j in matches:
        left_box = left_results['absolute_boxes'][i]
        right_box = right_results['absolute_boxes'][j]

        disparity, obj_x, obj_y = compute_disparity(left_box, right_box)
        depth = (FOCAL_LENGTH * BASELINE) / disparity  # Глубина в мм

        depths.append((obj_x, obj_y, depth))  # (X, Y, Depth)

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

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area) if (
                                                                                              boxA_area + boxB_area - intersection_area) > 0 else 0
    return iou


def match_boxes(left_results, right_results, iou_threshold=0.9):
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


# === 2. ЗАПУСК КАМЕР И ДЕТЕКЦИИ ===
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
            print(f"🔍 Matches found: {matches}")

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
