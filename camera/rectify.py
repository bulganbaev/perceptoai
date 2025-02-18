import cv2
import numpy as np

# Загружаем параметры калибровки
calib_data = np.load("calibration_data.npz")
mtxL, distL = calib_data["mtxL"], calib_data["distL"]
mtxR, distR = calib_data["mtxR"], calib_data["distR"]
R, T = calib_data["R"], calib_data["T"]

# Загружаем примеры изображений
imgL = cv2.imread("images/left/left_00.jpg")
imgR = cv2.imread("images/right/right_00.jpg")
height, width = imgL.shape[:2]

# Вычисляем матрицы ректификации
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, (width, height), R, T, alpha=0
)

# Создаем карты преобразования
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (width, height), cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (width, height), cv2.CV_16SC2)

# Применяем выравнивание
rectifiedL = cv2.remap(imgL, mapL1, mapL2, interpolation=cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, mapR1, mapR2, interpolation=cv2.INTER_LINEAR)

# Объединяем и показываем результат
rectified_pair = np.hstack((rectifiedL, rectifiedR))
cv2.imshow("Rectified Stereo Images", rectified_pair)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохраняем выровненные изображения
cv2.imwrite("images/left/left_rectified.jpg", rectifiedL)
cv2.imwrite("images/right/right_rectified.jpg", rectifiedR)
print("✅ Изображения выровнены и сохранены!")

np.savez("rectification_data.npz", Q=Q)
