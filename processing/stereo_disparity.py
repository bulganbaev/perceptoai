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

def compute_disparity_map(left_image, right_image):
    """
    Вычисление карты диспаритета с использованием StereoSGBM.

    Параметры:
        left_image  : Изображение с левой камеры (BGR)
        right_image : Изображение с правой камеры (BGR)

    Возвращает:
        disparity : Карта диспаритета (тип float32)
    """
    # Преобразование в оттенки серого
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Параметры StereoSGBM (при необходимости можно настроить)
    min_disp = 0
    num_disp = 16 * 8  # должно быть кратно 16
    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity


def filter_people(results):
    """
    Фильтрация детекций для выбранного класса.
    Здесь оставляем только объекты с class_id==2 (например, машина).
    При необходимости измените условие фильтрации.
    """
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


def compute_depth(left_results, right_results, matches, disparity_map):
    """
    Вычисление глубины для сопоставленных объектов с использованием карты диспаритета.

    Параметры:
        left_results  : Результаты детекции для левого изображения
        right_results : Результаты детекции для правого изображения
        matches       : Список пар индексов сопоставленных bbox (Hungarian Algorithm)
        disparity_map : Карта диспаритета, вычисленная из стереопары изображений

    Возвращает:
        Список кортежей (X, Y, final_depth, min_depth, max_depth) для каждого объекта.
    """
    global depth_history
    depths = {}

    for left_idx, right_idx in matches:
        left_box = left_results['absolute_boxes'][left_idx]
        right_box = right_results['absolute_boxes'][right_idx]

        # Используем координаты левого bbox для выборки диспаритета
        x1, y1, x2, y2 = left_box
        h, w = disparity_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Извлекаем значения диспаритета в пределах bbox
        box_disp = disparity_map[y1:y2, x1:x2]
        valid_disp = box_disp[box_disp > 0]  # Исключаем невалидные значения

        if valid_disp.size > 0:
            median_disp = np.median(valid_disp)
            min_disp = np.min(valid_disp)
            max_disp = np.max(valid_disp)
        else:
            # Если нет валидных значений в bbox, вычисляем диспаритет по центрам bbox
            center_L_x = (left_box[1] + left_box[3]) // 2
            center_R_x = (right_box[1] + right_box[3]) // 2
            median_disp = max(1, abs(center_L_x - center_R_x))
            min_disp = median_disp
            max_disp = median_disp

        # Вычисляем глубину: Depth = (focal_length * baseline) / disparity
        raw_depth = (FOCAL_LENGTH * BASELINE) / median_disp  # Глубина в мм

        # Применяем фильтр для сглаживания резких изменений глубины
        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)
        depth_history[obj_id].append(raw_depth)
        final_depth = np.median(depth_history[obj_id])

        # Определяем координаты центра bbox для отрисовки
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # Вычисляем диапазон глубины (при перерасчёте min/max по диспаритету)
        depth_min = (FOCAL_LENGTH * BASELINE) / max_disp
        depth_max = (FOCAL_LENGTH * BASELINE) / min_disp

        depths[obj_id] = (center_x, center_y, final_depth, depth_min, depth_max)

    return list(depths.values())


def match_boxes(left_results, right_results):
    """
    Сопоставление bbox между левым и правым изображениями по X-координатам с использованием алгоритма Хангариана.

    Возвращает:
        Список пар индексов (левый, правый)
    """
    left_boxes, right_boxes = left_results["absolute_boxes"], right_results["absolute_boxes"]
    if not left_boxes or not right_boxes:
        return []

    left_centers = np.array([(box[1] + box[3]) // 2 for box in left_boxes])
    right_centers = np.array([(box[1] + box[3]) // 2 for box in right_boxes])

    cost_matrix = np.abs(left_centers[:, None] - right_centers[None, :])
    left_indices, right_indices = linear_sum_assignment(cost_matrix)
    return [(l, r) for l, r in zip(left_indices, right_indices)]


def draw_boxes(image, results):
    """
    Отрисовка bounding boxes на изображении.
    """
    for (y1, x1, y2, x2), score in zip(results['absolute_boxes'], results['detection_scores']):
        label = f"Car ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_depth(image, depth_results):
    """
    Отрисовка информации о глубине (среднее значение и диапазон) на изображении.
    """
    for x, y, d, d_min, d_max in depth_results:
        text = f"{d:.1f}mm ({d_min:.1f}-{d_max:.1f}mm)"
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def choose_model():
    """
    Выбор модели для детекции.
    """
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


# === 4. ЗАПУСК КАМЕР И ДЕТЕКЦИИ ===
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
            # Вычисляем карту диспаритета для стереопары
            disparity_map = compute_disparity_map(frame_left, frame_right)

            # Выполняем детекцию объектов на обоих изображениях
            detections = proc.process([frame_left, frame_right])
            result_left = filter_people(detections[0])
            result_right = filter_people(detections[1])

            # Сопоставляем bbox между левым и правым изображениями
            matches = match_boxes(result_left, result_right)

            # Вычисляем глубину объектов, используя карту диспаритета
            depth_results = compute_depth(result_left, result_right, matches, disparity_map)

            # Отрисовка bounding boxes и информации о глубине
            processed_left = draw_boxes(frame_left, result_left)
            processed_left = draw_depth(processed_left, depth_results)

            # Объединяем левый и правый кадры для отображения
            combined = cv2.hconcat([processed_left, frame_right])
            cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stereo Depth", 1920, 1080)
            cv2.imshow("Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
