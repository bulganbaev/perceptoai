import cv2
import os
import time
from camera_driver import CameraDriver

# Создаем папки, если их нет
os.makedirs("images/left", exist_ok=True)
os.makedirs("images/right", exist_ok=True)

# Запускаем две камеры
cam0 = CameraDriver(camera_id=1)
cam1 = CameraDriver(camera_id=0)
cam0.start_camera()
cam1.start_camera()

image_count = 0

print("📸 Нажмите 's' для съемки, 'q' для выхода")
try:
    while True:
        frame0 = cam0.get_frame()
        frame1 = cam1.get_frame()

        if frame0 is not None and frame1 is not None:
            combined = cv2.hconcat([frame0, frame1])
            cv2.imshow("Stereo Capture", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename_left = f"images/left/left_{image_count:02d}.jpg"
            filename_right = f"images/right/right_{image_count:02d}.jpg"

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
cam0.stop_camera()
cam1.stop_camera()
cv2.destroyAllWindows()
print("📁 Все снимки сохранены в 'images/left' и 'images/right'")
