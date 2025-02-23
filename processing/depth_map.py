import cv2
import numpy as np
from cam.camera_driver import CameraDriver

# === 🔧 ПАРАМЕТРЫ STEREO SGBM ===
minDisparity = 0
numDisparities = 128  # Должно быть кратно 16
blockSize = 5
uniquenessRatio = 10
speckleWindowSize = 100
speckleRange = 32
disp12MaxDiff = 1

# === 🔬 НАСТРОЙКА STEREO SGBM ===
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff
)

# === 🛠️ НАСТРОЙКА ФИЛЬТРА WLS ===
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
stereo_right = cv2.ximgproc.createRightMatcher(stereo)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# === 🎥 ИНИЦИАЛИЗАЦИЯ КАМЕР ===
cam_left = CameraDriver(camera_id=0)
cam_right = CameraDriver(camera_id=1)

cam_left.start_camera()
cam_right.start_camera()

print("🎥 Запуск стереопотока с улучшенной картой глубины. Нажмите 'q' для выхода.")

try:
    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        if frame_left is not None and frame_right is not None:
            # === 📌 Преобразуем в Ч/Б для расчета disparity ===
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            # === 🏗️ Вычисление карты диспарити (Stereo SGBM) ===
            disp_left = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
            disp_right = stereo_right.compute(gray_right, gray_left).astype(np.float32) / 16.0

            # === 🛠 Применяем WLS-фильтр для сглаживания ===
            filtered_disp = wls_filter.filter(disp_left, gray_left, None, disp_right)

            # === 🔥 Нормализация карты глубины ===
            disp_norm = cv2.normalize(filtered_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            disp_norm = np.uint8(disp_norm)

            # === 🎨 Цветная карта глубины ===
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
