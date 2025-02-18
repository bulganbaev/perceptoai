import cv2
import numpy as np
import time
import open3d as o3d

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

# Преобразуем карту диспаритета в глубину
depth_map = cv2.reprojectImageTo3D(disparity, Q)

# Формируем облако точек
points = depth_map.reshape(-1, 3)
colors = cv2.cvtColor(cv2.imread("images/left/left_rectified.jpg"), cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

# Фильтруем точки с бесконечной глубиной
mask = (disparity > disparity.min())
points = points[mask.ravel()]
colors = colors[mask.ravel()]

# Создаем облако точек
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Останавливаем таймер
end_time = time.time()
elapsed_time = end_time - start_time

# Визуализация облака точек
o3d.visualization.draw_geometries([pcd])

# Сохраняем облако точек
o3d.io.write_point_cloud("point_cloud.ply", pcd)

# Выбираем центральную точку для измерения расстояния
h, w = disparity.shape
center_distance = depth_map[h // 2, w // 2, 2]

print("✅ Облако точек сохранено в point_cloud.ply")
print(f"⏳ Оптимизированное время расчета: {elapsed_time:.4f} секунд")
print(f"📏 Расстояние до объекта в центре кадра: {center_distance:.2f} мм")