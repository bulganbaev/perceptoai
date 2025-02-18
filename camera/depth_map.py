import cv2
import numpy as np
import time

# Загружаем ректифицированные изображения
imgL = cv2.imread("images/left/left_rectified.jpg", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("images/right/right_rectified.jpg", cv2.IMREAD_GRAYSCALE)

# Проверяем, что изображения загружены
if imgL is None or imgR is None:
    print("Ошибка: ректифицированные изображения не найдены!")
    exit()

# Запускаем таймер
start_time = time.time()

# Оптимизированные параметры SGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=32,  # Уменьшено для ускорения
    blockSize=5,  # Меньше размер блока - быстрее
    uniquenessRatio=5,
    speckleWindowSize=50,
    speckleRange=16,
    disp12MaxDiff=0,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2
)

# Вычисляем карту диспаритета
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Останавливаем таймер
end_time = time.time()
elapsed_time = end_time - start_time

# Нормализуем для визуализации
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Показываем результат
cv2.imshow("Depth Map", disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохраняем карту глубины
cv2.imwrite("depth_map.jpg", disparity_normalized)
print("✅ Карта глубины сохранена в depth_map.jpg")
print(f"⏳ Оптимизированное время расчета: {elapsed_time:.4f} секунд")
