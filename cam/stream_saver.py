import time
import cv2
import os
from datetime import datetime
from camera_driver import StereoCameraSystem

time.sleep(10)
s
# Создаем папки для сохранения снимков
os.makedirs("data/images/left", exist_ok=True)
os.makedirs("data/images/right", exist_ok=True)

# Запускаем стерео-камеру
stereoCam = StereoCameraSystem()
stereoCam.start()

print("📸 Автоматический режим: сохранение кадров каждую секунду (нажмите Ctrl+C для выхода)")

try:
    while True:
        frame0, frame1 = stereoCam.get_synchronized_frames()

        if frame0 is not None and frame1 is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename_left = f"data/images/left/left_{timestamp}.jpg"
            filename_right = f"data/images/right/right_{timestamp}.jpg"

            cv2.imwrite(filename_left, frame0)
            cv2.imwrite(filename_right, frame1)

            print(f"✅ Снимки сохранены: {filename_left}, {filename_right}")

        time.sleep(1)  # Интервал между снимками (можно изменить)

except KeyboardInterrupt:
    print("\n🛑 Остановка скрипта...")

# Останавливаем камеры
stereoCam.stop()
print("📁 Все снимки сохранены в 'data/images/left' и 'data/images/right'")
