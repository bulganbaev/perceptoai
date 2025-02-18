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

# Загружаем параметры калибровки для перевода диспаритета в глубину
calib_data = np.load("calibration_data.npz")
Q = calib_data.get("Q")  # Матрица реконструкции 3D
if Q is None:
    print("Ошибка: Матрица Q не найдена в calibration_data.npz!")
    exit()

# Запускаем таймер
start_time = time.time()

# Оптимизированные параметры StereoBM (более быстрый алгоритм)
stereo = cv2.StereoBM_create(
    numDisparities=32,  # Должно быть кратно 16
    blockSize=9  # Баланс между качеством и скоростью
)

# Вычисляем карту диспаритета
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Останавливаем таймер
end_time = time.time()
elapsed_time = end_time - start_time

# Преобразуем карту диспаритета в глубину
depth_map = cv2.reprojectImageTo3D(disparity, Q)
depth_values = depth_map[:, :, 2]  # Z-координата

# Выбираем центральную точку для измерения расстояния
h, w = disparity.shape
center_distance = depth_values[h // 2, w // 2]

# Нормализуем карту диспаритета для визуализации
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Показываем карту диспаритета
cv2.imshow("Disparity Map", disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохраняем карту диспаритета
cv2.imwrite("disparity_map.jpg", disparity_normalized)
print("✅ Карта диспаритета сохранена в disparity_map.jpg")
print(f"⏳ Время расчета: {elapsed_time:.4f} секунд")
print(f"📏 Расстояние до объекта в центре кадра: {center_distance:.2f} мм")