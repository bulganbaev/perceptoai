import cv2
import os
import time
from camera_driver import StereoCameraSystem

# Создаем папки, если их нет
os.makedirs("data/images/left", exist_ok=True)
os.makedirs("data/images/right", exist_ok=True)

# Запускаем две камеры
stereoCam = StereoCameraSystem()
stereoCam.start()

image_count = 0

print("📸 Нажмите 's' для съемки, 'q' для выхода")
try:
    while True:
        frame0, frame1 = stereoCam.get_synchronized_frames()

        if frame0 is not None and frame1 is not None:
            combined = cv2.hconcat([frame0, frame1])
            cv2.namedWindow("Dual Cameras", cv2.WINDOW_NORMAL)  # Делаем окно изменяемым
            cv2.resizeWindow("Dual Cameras", 1024, 768)
            cv2.imshow("Dual Cameras", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename_left = f"data/images/left/left_{image_count:02d}.jpg"
            filename_right = f"data/images/right/right_{image_count:02d}.jpg"

            cv2.imwrite(filename_left, frame0)
            cv2.imwrite(filename_right, frame1)

            print(f"✅ Снимок сохранен: {filename_left}, {filename_right}")
            image_count += 1
            time.sleep(0.5)  # Чтобы избежать двойного снимка

        elif key == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Останавливаем камеры
stereoCam.stop()
cv2.destroyAllWindows()
print("📁 Все снимки сохранены в 'images/left' и 'images/right'")
