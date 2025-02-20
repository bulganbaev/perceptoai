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
            configure_params = hp.ConfigureParams.create_from_hef(self.hef, interface=hp.HailoStreamInterface.PCIe)
            self.network_groups = self.vdevice.configure(self.hef, configure_params)
            self.configured_network = self.network_groups[0]  # Берём первую (и единственную) сеть

            # Создаём параметры потоков через make_from_network_group
            self.input_vstreams_params = hp.InputVStreamParams.make_from_network_group(self.configured_network)
            self.output_vstreams_params = hp.OutputVStreamParams.make_from_network_group(self.configured_network,
                                                                                         format_type=hp.FormatType.UINT8)

            # Создаём потоки для инференса
            self.infer_vstreams = hp.InferVStreams(self.configured_network, self.input_vstreams_params,
                                                   self.output_vstreams_params)

            print("✅ Hailo-8 успешно подключен. Потоки инференса созданы.")
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
            # Преобразование к размеру модели
            imgL_resized = cv2.resize(imgL, (1232, 368))
            imgR_resized = cv2.resize(imgR, (1232, 368))
            input_tensor = np.stack((imgL_resized, imgR_resized), axis=0).astype(np.float32)

            # Преобразуем данные в uint8
            input_tensor = (input_tensor * 255).astype(np.uint8)

            # Проверка соответствия размеров
            print("Expected input shape:", self.input_vstreams_params["stereonet/input_layer1"].shape)
            print("Actual input shape:", input_tensor[0].shape)

            # Запуск инференса на Hailo-8
            with self.infer_vstreams as infer_pipeline:
                with self.configured_network.activate():
                    output_data = infer_pipeline.infer(
                        {"stereonet/input_layer1": input_tensor[0], "stereonet/input_layer2": input_tensor[1]})
                    disparity = output_data["stereonet/conv53"]

            # Масштабируем disparity обратно к оригинальному разрешению
            disparity = cv2.resize(disparity, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            disparity = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        # Вычисляем depth map
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
