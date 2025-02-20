import cv2
import numpy as np
import os
import time
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

        # Вычисляем матрицы ремапинга
        img_width, img_height = 1232, 368
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtxL, self.distL, self.mtxR, self.distR, (img_width, img_height), self.R, self.T, alpha=1)

        self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(self.mtxL, self.distL, self.R1, self.P1,
                                                             (img_width, img_height), cv2.CV_16SC2)
        self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.mtxR, self.distR, self.R2, self.P2,
                                                             (img_width, img_height), cv2.CV_16SC2)

        self.use_hailo = use_hailo
        if use_hailo:
            self.vdevice = hp.VDevice()
            self.hef = hp.HEF(hef_path)
            configure_params = hp.ConfigureParams.create_from_hef(self.hef, interface=hp.HailoStreamInterface.PCIe)
            self.network_groups = self.vdevice.configure(self.hef, configure_params)
            self.configured_network = self.network_groups[0]

            self.input_vstream_infos = self.configured_network.get_input_vstream_infos()
            self.output_vstream_infos = self.configured_network.get_output_vstream_infos()

            self.input_vstreams_params = hp.InputVStreamParams.make_from_network_group(self.configured_network)
            self.output_vstreams_params = hp.OutputVStreamParams.make_from_network_group(self.configured_network,
                                                                                         format_type=hp.FormatType.UINT8)

            self.infer_vstreams = hp.InferVStreams(self.configured_network, self.input_vstreams_params,
                                                   self.output_vstreams_params)

            print("✅ Hailo-8 успешно подключен. Потоки инференса созданы.")
        else:
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=96,
                blockSize=9,
                P1=8 * 3 * 9 ** 2,
                P2=32 * 3 * 9 ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32
            )

    def compute_depth(self, imgL_path, imgR_path, save_path="data/images/depth_map.png"):
        start_time = time.time()

        imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)

        if imgL is None or imgR is None:
            raise ValueError("Ошибка загрузки изображений! Проверьте пути.")

        imgL_rect = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
        imgR_rect = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

        if self.use_hailo:
            imgL_resized = np.ascontiguousarray(
                cv2.cvtColor(cv2.resize(imgL_rect, (1232, 368)), cv2.COLOR_GRAY2RGB).astype(np.uint8)).reshape(1, 368,
                                                                                                               1232, 3)
            imgR_resized = np.ascontiguousarray(
                cv2.cvtColor(cv2.resize(imgR_rect, (1232, 368)), cv2.COLOR_GRAY2RGB).astype(np.uint8)).reshape(1, 368,
                                                                                                               1232, 3)

            input_data = {"stereonet/input_layer1": imgL_resized, "stereonet/input_layer2": imgR_resized}

            with self.infer_vstreams as infer_pipeline:
                with self.configured_network.activate():
                    output_data = infer_pipeline.infer(input_data)
                    disparity = output_data.get("stereonet/conv53")

            if disparity is None or disparity.size == 0:
                raise ValueError("Ошибка: disparity пуст или не получен от модели!")

            disparity = np.squeeze(disparity)
            disparity = cv2.resize(disparity, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            disparity = self.stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0

        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5)

        focal_length = self.mtxL[0, 0]
        depth_map = (focal_length * self.baseline) / (disparity + 1e-6)

        depth_visual = cv2.applyColorMap(cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                                         cv2.COLORMAP_JET)
        cv2.imwrite(save_path, depth_visual)

        elapsed_time = time.time() - start_time
        print(f"⏱ Время выполнения: {elapsed_time:.4f} сек")
        return depth_visual


if __name__ == "__main__":
    depth_estimator = DepthEstimator(use_hailo=True)
    depth_map = depth_estimator.compute_depth("data/images/left/left_00.jpg", "data/images/right/right_00.jpg")
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✅ Карта глубины сохранена в data/images/depth_map.png")
