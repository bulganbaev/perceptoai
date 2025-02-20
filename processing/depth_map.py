import cv2
import numpy as np
import os
import hailo_platform as hp


class DepthEstimator:
    def __init__(self, calib_path="data/calibration/calibration_data.npz", use_hailo=True,
                 hef_path="data/models/stereonet.hef"):
        # Загружаем параметры калибровки
        calib_data = np.load(calib_path)
        self.mtxL, self.distL = calib_data["mtxL"], calib_data["distL"]
        self.mtxR, self.distR = calib_data["mtxR"], calib_data["distR"]
        self.R, self.T = calib_data["R"], calib_data["T"]

        # Вычисляем baseline (расстояние между камерами)
        self.baseline = np.linalg.norm(self.T)

        self.use_hailo = use_hailo
        if use_hailo:
            self.device = hp.Device()
            self.hef = hp.HEF(hef_path)
            self.network_group = self.device.configure(self.hef)
            print("✅ Hailo-8 успешно подключен. Модель загружена.")
            print("ℹ️ Информация о сети:")
            print(self.network_group.get_network_info())
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


if __name__ == "__main__":
    depth_estimator = DepthEstimator(use_hailo=True)  # Включаем Hailo-8
    print("🔄 Проверяем загрузку модели на Hailo-8...")
