#!/usr/bin/env python3
"""
Основной скрипт для запуска стереопотока с детекцией объектов с использованием HailoInference.
"""

import cv2
import os
import numpy as np
from cam.camera_driver import StereoCameraSystem
from processing.hailo_detection import HailoInference, Processor


def choose_model():
    """
    Позволяет выбрать модель для детекции из директории data/models.

    Возвращает:
        str: Полный путь к выбранной модели (.hef файл).
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


def main():
    # Инициализация модели и инференса
    model_path = choose_model()
    inference = HailoInference(model_path)
    processor = Processor(inference, conf=0.5)

    # Инициализация стереокамеры
    stereo = StereoCameraSystem()
    stereo.start()

    print("🎥 Запуск стереопотока. Нажмите 'q' для выхода.")

    try:
        while True:
            frame_left, frame_right = stereo.get_synchronized_frames()

            if frame_left is not None and frame_right is not None:
                # Выполняем детекцию объектов для левого и правого кадров
                segmentations = processor.process([frame_left, frame_right])

                # Получаем маски для левого изображения
                left_masks = segmentations[0].get('absolute_masks', [])

                # Создаем пустую маску для левого изображения
                left_mask_overlay = np.zeros_like(frame_left)

                # Накладываем все найденные маски, объединяя их через np.maximum
                for mask in left_masks:
                    left_mask_overlay[:, :, 2] = np.maximum(left_mask_overlay[:, :, 2], mask * 255)

                # Объединяем оригинальное левое изображение с наложенной маской
                left_blended = cv2.addWeighted(frame_left, 0.7, left_mask_overlay, 0.3, 0)

                # Приводим изображение к разрешению Full HD (1920x1080)
                combined_resized = cv2.resize(left_blended, (1920, 1080))

                # Отображаем результат
                cv2.imshow("Stereo Segmentation", combined_resized)

            # Выход по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("⏹️ Остановка потока...")

    # Очистка ресурсов
    stereo.stop()
    cv2.destroyAllWindows()
    print("✅ Поток завершён.")


if __name__ == '__main__':
    main()
