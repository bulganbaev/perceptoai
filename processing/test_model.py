import cv2
import os
import numpy as np
from cam.camera_driver import StereoCameraSystem
from processing.hailo_segmentation import HailoSegmentation, Processor

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
inf = HailoSegmentation(model_path)
proc = Processor(inf)
stereo = StereoCameraSystem()
stereo.start()

print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left, frame_right = stereo.get_synchronized_frames()

        if frame_left is not None and frame_right is not None:
            # Выполняем сегментацию
            segmentations = proc.process([frame_left, frame_right])

            # Получаем маски для левого изображения
            left_mask = segmentations[0]

            # Создаем пустую маску для наложения
            left_mask_overlay = np.zeros_like(frame_left)
            left_mask_overlay[:, :, 2] = (left_mask * 255).astype(np.uint8)  # Добавляем в синий канал

            # Объединяем оригинальное левое изображение с маской
            left_blended = cv2.addWeighted(frame_left, 0.7, left_mask_overlay, 0.3, 0)

            # Приводим к Full HD (1920x1080)
            combined_resized = cv2.resize(left_blended, (1920, 1080))

            # Показываем результат
            cv2.imshow("Stereo Segmentation", combined_resized)

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# === ОЧИСТКА РЕСУРСОВ ===
stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
