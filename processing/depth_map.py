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
            self.vdevice = hp.VDevice()
            self.hef = hp.HEF(hef_path)
            self.network_groups = self.vdevice.configure(self.hef)
            self.configured_network = self.network_groups[0]  # Берём первую (и единственную) сеть

            # Создаём потоки
            self.input_vstreams = hp.InputVStreams(self.configured_network,
                                                   ["stereonet/input_layer1", "stereonet/input_layer2"])
            self.output_vstreams = hp.OutputVStreams(self.configured_network, ["stereonet/conv53"])

            print("✅ Hailo-8 успешно подключен. Потоки созданы.")
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
    print("🔄 Потоки инференса успешно созданы! Готовимся к тесту...")
