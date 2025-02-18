import cv2
import numpy as np
import glob

# Настройки шахматной доски
CHESSBOARD_SIZE = (9, 6)  # Количество внутренних углов
SQUARE_SIZE = 25  # Размер клетки в мм

# Создаем массив реальных координат точек
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Списки для хранения точек
objpoints = []  # 3D точки в реальном мире
imgpoints_left = []  # 2D точки для левой камеры
imgpoints_right = []  # 2D точки для правой камеры

# Загружаем изображения
left_images = sorted(glob.glob("images/left/*.jpg"))
right_images = sorted(glob.glob("images/right/*.jpg"))

assert len(left_images) == len(right_images), "Количество изображений должно совпадать!"

# Обрабатываем каждую пару изображений
for left_img, right_img in zip(left_images, right_images):
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

# Калибруем каждую камеру отдельно
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# Стерео-калибровка (поиск взаимного положения камер)
(retS, mtxL, distL, mtxR, distR, R, T, E, F) = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Сохраняем параметры
np.savez("calibration_data.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print("✅ Калибровка завершена! Параметры сохранены в calibration_data.npz")
