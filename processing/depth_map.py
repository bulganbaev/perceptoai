import cv2
import numpy as np
import os
import hailo


class DepthEstimator:
    def __init__(self, calib_path="data/calibration/calibration_data.npz", use_hailo=True,
                 hef_path="/mnt/data/stereonet.hef"):
        # Загружаем параметры калибровки
        calib_data = np.load(calib_path)
        self.mtxL, self.distL = calib_data["mtxL"], calib_data["distL"]
        self.mtxR, self.distR = calib_data["mtxR"], calib_data["distR"]
        self.R, self.T = calib_data["R"], calib_data["T"]

        # Вычисляем baseline (расстояние между камерами)
        self.baseline = np.linalg.norm(self.T)

        self.use_hailo = use_hailo
        if use_hailo:
            self.device = hailo.HailoRTDevice()
            self.network_group = self.device.create_network_group(hef_path)
            self.input_shape = (368, 1232)  # Оптимальное разрешение входа для StereoNet
        else:
            # Создаем SGBM стерео-пару
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=96,  # Должно быть кратно 16
                blockSize=9,
                P1=8 * 3 * 9 ** 2,
                P2=32 * 3 * 9 ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32
            )

    def compute_depth(self, imgL_path, imgR_path, save_path="data/images/depth_map.png"):
        # Загружаем изображения
        imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)

        if imgL is None or imgR is None:
            raise ValueError("Ошибка загрузки изображений! Проверьте пути.")

        if self.use_hailo:
            # Подготовка изображений для Hailo (приводим к 368x1232)
            imgL_resized = cv2.resize(imgL, self.input_shape[::-1])
            imgR_resized = cv2.resize(imgR, self.input_shape[::-1])
            input_tensor = np.stack((imgL_resized, imgR_resized), axis=-1).astype(np.float32) / 255.0

            # Запуск модели StereoNet на Hailo
            self.network_group.execute(input_tensor)
            output_tensors = self.network_group.get_all_output_tensors()
            disparity = output_tensors[0]

            # Быстрое масштабирование disparity до оригинального размера
            disparity = cv2.resize(disparity, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            disparity = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        focal_length = self.mtxL[0, 0]  # Фокусное расстояние из матрицы камеры
        depth_map = (focal_length * self.baseline) / (disparity + 1e-6)  # +1e-6 для избегания деления на 0

        # Нормализация и визуализация
        depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(save_path, depth_visual)
        return depth_visual


if __name__ == "__main__":
    depth_estimator = DepthEstimator(use_hailo=True)  # Включаем Hailo-8
    depth_map = depth_estimator.compute_depth("data/images/left/left_00.jpg", "data/images/right/right_00.jpg")

    # Отображение результата
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✅ Карта глубины сохранена в data/images/depth_map.png")
