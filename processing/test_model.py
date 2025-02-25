import cv2
import os
import numpy as np
from cam.camera_driver import StereoCameraSystem
from processing.hailo_detection import HailoInference, HailoSegmentation

def choose_model():
    """
    Выбор модели для сегментации.
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
hailo_model = HailoInference(model_path)
segmentation = HailoSegmentation(hailo_model)

stereo = StereoCameraSystem()
stereo.start()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left, frame_right = stereo.get_synchronized_frames()

        if frame_left is not None and frame_right is not None:
            # Выполняем сегментацию
            segmentations = segmentation.run_inference(frame_left)


        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# === ОЧИСТКА РЕСУРСОВ ===
stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
