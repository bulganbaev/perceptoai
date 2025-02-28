import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from cam.camera_driver import StereoCameraSystem
from processing.hailo.hailo_detection import HailoInference, Processor

# === 1. ЗАГРУЗКА КАЛИБРОВКИ ===
calib_data = np.load("data/calibration/calibration_data.npz")
mtxL, distL = calib_data["mtxL"], calib_data["distL"]
mtxR, distR = calib_data["mtxR"], calib_data["distR"]
R, T = calib_data["R"], calib_data["T"]

BASELINE = abs(T[0][0])  # Расстояние между камерами (мм)

# === 2. ФИЛЬТР ДЛЯ СТАБИЛЬНОГО DEPTH ===
depth_history = {}
DEPTH_FILTER_SIZE = 5  # Размер скользящего окна

# === 3. НАСТРОЙКА ВИДЕОЗАПИСИ ===
out_file = "output.mp4"
frame_size = (1024,768)  # Разрешение Full HD
fps = 30  # Частота кадров
fourcc = cv2.VideoWriter_fourcc(*"H264")  # Кодек для сохранения видео
out = cv2.VideoWriter(out_file, fourcc, fps, frame_size)


# === 3. ФУНКЦИИ ===

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
        if class_id == 39:
            filtered_boxes.append(results['absolute_boxes'][i])
            filtered_scores.append(results['detection_scores'][i])
            filtered_classes.append(class_id)
    if not filtered_boxes:
        print("Лог: В текущем кадре не найдено объектов машины (class_id==2).")
    results.update({
        'absolute_boxes': filtered_boxes,
        'detection_classes': filtered_classes,
        'detection_scores': filtered_scores
    })
    return results


def compute_depth_rectified(left_results, right_results, matches, R1, R2, P1, P2):
    """
    Вычисление глубины для сопоставленных объектов с использованием ректификации центра bbox.
    Для каждого найденного объекта вычисляется центр bbox, затем точки ректифицируются,
    и вычисляется disparity как разница по X между ректифицированными точками.

    Параметры:
        left_results  : Результаты детекции для левого изображения.
        right_results : Результаты детекции для правого изображения.
        matches       : Список пар индексов сопоставленных bbox.
        R1, R2        : Ректификационные матрицы для левого и правого камер.
        P1, P2        : Новые проекционные матрицы для левого и правого камер.

    Возвращает:
        Список кортежей (center_x, center_y, final_depth) для каждого объекта.
    """
    global depth_history
    depths = {}
    if not matches:
        print("Лог: Не удалось сопоставить объекты между изображениями.")
    for left_idx, right_idx in matches:
        left_box = left_results['absolute_boxes'][left_idx]
        right_box = right_results['absolute_boxes'][right_idx]

        # Вычисляем центры bbox. Предполагаем формат bbox: [y1, x1, y2, x2]
        center_left_x = (left_box[1] + left_box[3]) // 2
        center_left_y = (left_box[0] + left_box[2]) // 2
        center_right_x = (right_box[1] + right_box[3]) // 2
        center_right_y = (right_box[0] + right_box[2]) // 2

        # Формируем точки для ректификации
        pt_left = np.array([[[center_left_x, center_left_y]]], dtype=np.float32)
        pt_right = np.array([[[center_right_x, center_right_y]]], dtype=np.float32)

        # Ректификация точек
        rect_left = cv2.undistortPoints(pt_left, mtxL, distL, R=R1, P=P1)
        rect_right = cv2.undistortPoints(pt_right, mtxR, distR, R=R2, P=P2)

        # Извлекаем координаты X ректифицированных точек
        x_left = rect_left[0, 0, 0]
        x_right = rect_right[0, 0, 0]

        # Вычисляем disparity
        disparity = abs(x_left - x_right)
        if disparity < 1:
            disparity = 1  # избегаем деления на 0

        # Используем фокусное расстояние из P1 (элемент [0,0])
        focal_rect = P1[0, 0]
        # Вычисляем глубину: Depth = (focal * BASELINE) / disparity
        raw_depth = (focal_rect * BASELINE) / disparity

        # Фильтрация глубины для сглаживания
        obj_id = left_idx
        if obj_id not in depth_history:
            depth_history[obj_id] = deque(maxlen=DEPTH_FILTER_SIZE)
        depth_history[obj_id].append(raw_depth)
        final_depth = np.median(depth_history[obj_id])

        depths[obj_id] = (center_left_x, center_left_y, final_depth)
    if not depths:
        print("Лог: Глубина не вычислена, так как сопоставленных объектов нет.")
    return list(depths.values())


def match_boxes(left_results, right_results):
    """
    Сопоставление bbox между левым и правым изображениями по X-координатам с использованием алгоритма Хангариана.

    Возвращает:
        Список пар индексов (левый, правый).
    """
    left_boxes, right_boxes = left_results["absolute_boxes"], right_results["absolute_boxes"]
    if not left_boxes:
        print("Лог: В левом изображении не найдено bounding boxes.")
    if not right_boxes:
        print("Лог: В правом изображении не найдено bounding boxes.")
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
        label = f"Bottle ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def draw_depth(image, results):
    """
    Отрисовка информации о глубине (среднее значение) на изображении.
    """
    for x, y, depth in results:
        text = f"{depth:.1f} mm"
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def draw_status_text(image, text):
    """
    Отображает текст в левом верхнем углу кадра.

    Параметры:
        image: Кадр, на который нужно нанести текст.
        text: Текст, который нужно вывести.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, 50)  # Координаты для верхнего левого угла
    font_scale = 1  # Размер шрифта
    font_color = (0, 0, 255)  # Красный цвет (BGR)
    thickness = 2  # Толщина линии
    cv2.putText(image, text, position, font, font_scale, font_color, thickness)
    return image

def choose_model():
    """
    Выбор модели для детекции.
    """
    models_dir = "data/models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".hef")]
    if not model_files:
        print("Лог: Модели не найдены в директории", models_dir)
        exit(1)
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

# Вычисление ректификационных параметров один раз (при первом кадре)
rect_params_computed = False
R1 = R2 = P1 = P2 = None

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")
try:
    while True:
        frame_left, frame_right = stereo.get_synchronized_frames()
        if frame_left is not None and frame_right is not None:
            if not rect_params_computed:
                h, w = frame_left.shape[:2]
                # Вычисляем ректификационные матрицы для камер
                R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                    mtxL, distL, mtxR, distR, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY
                )
                rect_params_computed = True

            # Выполняем детекцию объектов на обоих изображениях
            detections = proc.process([frame_left, frame_right])
            result_left = filter_people(detections[0])
            result_right = filter_people(detections[1])

            status = "FOLLOWING"
            if not result_left['absolute_boxes']:
                print("Лог: В левом изображении не найдено объектов машины.")
                status = "SEARCHING"
            if not result_right['absolute_boxes']:
                print("Лог: В правом изображении не найдено объектов машины.")
                status = "SEARCHING"
            # Сопоставляем bbox между левым и правым изображениями
            matches = match_boxes(result_left, result_right)

            # Вычисляем глубину с использованием ректификации центра bbox
            depth_results = compute_depth_rectified(result_left, result_right, matches, R1, R2, P1, P2)

            # Отрисовка bounding boxes и информации о глубине на левом изображении
            processed_left = draw_boxes(frame_left, result_left)
            processed_left = draw_depth(processed_left, depth_results)
            processed_right = draw_status_text(draw_boxes(frame_right, result_right), status)
            print(processed_left.shape, processed_right.shape)
            # Объединяем левый и правый кадры для отображения
            combined = cv2.hconcat([processed_left, processed_right])
            out.write(combined)  # Запись кадра в видеофайл

            cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stereo Depth", 1024, 768)
            cv2.imshow("Stereo Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

stereo.stop()
out.release()  # Закрытие видеозаписи

cv2.destroyAllWindows()
print("✅ Поток завершён.")
