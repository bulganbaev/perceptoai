import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from cam.camera_driver import StereoCameraSystem
from processing.hailo_detection import HailoInference, Processor


def filter_people(results):
    """
    Фильтрация детекций для выбранного класса.
    Здесь оставляем только объекты с class_id==0 (например, человек).
    """
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []

    for i, class_id in enumerate(results['detection_classes']):
        if class_id == 0:
            filtered_boxes.append(results['absolute_boxes'][i])
            filtered_scores.append(results['detection_scores'][i])
            filtered_classes.append(class_id)

    if not filtered_boxes:
        print("Лог: В текущем кадре не найдено объектов класса 0 (человек).")

    results.update({
        'absolute_boxes': filtered_boxes,
        'detection_classes': filtered_classes,
        'detection_scores': filtered_scores
    })
    return results


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


# === 1. ИНИЦИАЛИЗАЦИЯ ===
model_path = choose_model()
inf = HailoInference(model_path)
proc = Processor(inf, conf=0.5)
stereo = StereoCameraSystem()
stereo.start()

# Вычисление ректификационных параметров один раз (при первом кадре)
rect_params_computed = False
stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=9,
    P1=8 * 3 * 9 ** 2,
    P2=32 * 3 * 9 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left, frame_right = stereo.get_synchronized_frames()

        if frame_left is not None and frame_right is not None:
            # Детекция объектов на изображениях
            segmentations = proc.process([frame_left, frame_right])

            # Фильтрация только объектов нужного класса (например, человек)
            # segmentations = [filter_people(result) for result in segmentations]

            # Отображение результатов
            for i, frame in enumerate([frame_left, frame_right]):
                for box in segmentations[i]['absolute_boxes']:
                    y1, x1, y2, x2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for mask in segmentations[i].get('absolute_masks', []):
                    mask_overlay = np.zeros_like(frame)
                    mask_overlay[:, :, 2] = mask * 255  # Синий канал для маски
                    blended = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
                    cv2.imshow(f"Segmentation {i}", blended)

            # === РЕКТИФИКАЦИЯ (Один раз) ===
            if not rect_params_computed:
                print("⚙️ Вычисление ректификационных параметров...")
                R1, R2, P1, P2, Q = stereo.compute_rectification_maps()
                rect_params_computed = True

            # === ВЫЧИСЛЕНИЕ ДИСПАРАТНОСТИ ===
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            disparity = stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

            # Маска для фильтрации невалидных значений
            valid_mask = disparity > 0

            # === ОЦЕНКА ГЛУБИНЫ (примерная) ===
            focal_length = P1[0, 0]  # Фокусное расстояние из матрицы P1
            baseline = 0.1  # Базис (расстояние между камерами в метрах)
            depth_map = np.zeros_like(disparity, dtype=np.float32)
            depth_map[valid_mask] = (focal_length * baseline) / (disparity[valid_mask] + 1e-6)

            # === ВИЗУАЛИЗАЦИЯ ГЛУБИНЫ ===
            depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

            cv2.imshow("Depth Map", depth_visual)
            cv2.imshow("Disparity", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

        # Выход при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# === ОЧИСТКА РЕСУРСОВ ===
stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
