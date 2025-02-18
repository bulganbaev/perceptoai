import cv2
import numpy as np
import glob

# Настройки шахматной доски
CHESSBOARD_SIZE = (10, 7)  # Количество внутренних углов
SQUARE_SIZE = 25  # Размер клетки в мм

# Создаем массив реальных координат точек
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Списки для хранения точек
objpoints = []  # 3D точки в реальном мире
imgpoints_left = []  # 2D точки для левой камеры
imgpoints_right = []  # 2D точки для правой камеры

# Загружаем изображения
left_images = sorted(glob.glob("../data/images/left/*.jpg"))
right_images = sorted(glob.glob("../data/images/right/*.jpg"))

assert len(left_images) == len(right_images), "Количество изображений должно совпадать!"
print(f"📸 Найдено {len(left_images)} пар изображений для калибровки")

# Обрабатываем каждую пару изображений
for i, (left_img, right_img) in enumerate(zip(left_images, right_images)):
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Ищем шахматную доску
    retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        print(f"✅ [{i+1}/{len(left_images)}] Шахматная доска найдена на обоих изображениях")
    else:
        print(f"❌ [{i+1}/{len(left_images)}] Шахматная доска не найдена, пропускаем")

# Калибруем каждую камеру отдельно
print("📏 Калибровка левой камеры...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
print(f"🎯 Лучшая средняя ошибка RMSE для левой камеры: {retL:.6f}")

print("📏 Калибровка правой камеры...")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
print(f"🎯 Лучшая средняя ошибка RMSE для правой камеры: {retR:.6f}")

# Стерео-калибровка (поиск взаимного положения камер)
print("🔄 Запуск стерео-калибровки...")
(retS, mtxL, distL, mtxR, distR, R, T, E, F) = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)
print(f"🔍 Стерео-калибровка завершена. Средняя ошибка RMSE: {retS:.6f}")

# Вычисляем baseline
baseline = np.linalg.norm(T)
print(f"📏 Baseline (расстояние между камерами): {baseline:.2f} мм")

# Вывод матрицы трансформации
print("🌀 Матрица поворота (R):\n", R)
print("🚀 Вектор трансляции (T):\n", T)

# Сохраняем параметры
np.savez("data/calibration/calibration_data.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print("✅ Калибровка завершена! Параметры сохранены в calibration_data.npz")
