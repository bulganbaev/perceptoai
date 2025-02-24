import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from cam.camera_driver import StereoCameraSystem
from processing.hailo_segmentation import HailoSegmentation, ProcessorSegmentation


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
        if class_id == 0:
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
inf = HailoSegmentation(model_path)
proc = ProcessorSegmentation()
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
            # Выполняем детекцию объектов на обоих изображениях
            detections = proc.process([frame_left, frame_right])

            cv2.imwrite("segmentation_left.png", detections[0])


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
