import cv2
import numpy as np
from cam.camera_driver import CameraDriver

# === 🔧 ПАРАМЕТРЫ STEREO BM ===
numDisparities = 64  # Должно быть кратно 16
blockSize = 15        # Размер блока для сравнения

# === 🎥 ИНИЦИАЛИЗАЦИЯ КАМЕР ===
cam_left = CameraDriver(camera_id=0)
cam_right = CameraDriver(camera_id=1)

cam_left.start_camera()
cam_right.start_camera()

print("🎥 Запуск стереопотока с картой глубины. Нажмите 'q' для выхода.")

# === 🔬 НАСТРОЙКА STEREO BM ===
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

try:
    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            # === 📌 Преобразуем в Ч/Б для расчета disparity ===
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            # === 🏗️ Вычисление карты диспарити ===
            disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

            # === 🎨 Нормализация диспарити в 8-битное изображение ===
            disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            disp_norm = np.uint8(disp_norm)

            # === 🔥 Накладываем цветовую карту для удобства восприятия ===
            depth_colormap = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

            # === 📺 Вывод результата ===
            cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth Map", 1920, 1080)
            cv2.imshow("Depth Map", depth_colormap)

        # === ⏹️ Выход по 'q' ===
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹️ Остановка потока...")

# === Завершаем работу камер ===
cam_left.stop_camera()
cam_right.stop_camera()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
