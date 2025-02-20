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

        # Размер входа модели
        self.model_w, self.model_h = 1232, 368  # Размер, требуемый StereoNet

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

    def preprocess_stereo(self, imgL, imgR):
        """
        Масштабирует изображения и центрирует их на черном фоне.
        """
        def resize_and_pad(image):
            img_h, img_w, _ = image.shape
            scale = min(self.model_w / img_w, self.model_h / img_h)
            new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
            image_resized = cv2.resize(image, (new_img_w, new_img_h))

            padded_image = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)
            pasted_w = (self.model_w - new_img_w) // 2
            pasted_h = (self.model_h - new_img_h) // 2
            padded_image[pasted_h:pasted_h + new_img_h, pasted_w:pasted_w + new_img_w, :] = image_resized
            return padded_image

        imgL_padded = resize_and_pad(imgL)
        imgR_padded = resize_and_pad(imgR)

        return imgL_padded, imgR_padded

    def compute_depth(self, imgL_path, imgR_path, save_path="data/images/depth_map.png"):
        start_time = time.time()

        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)

        if imgL is None or imgR is None:
            raise ValueError("Ошибка загрузки изображений! Проверьте пути.")

        # Применяем корректный ресайз с центрированием
        imgL_padded, imgR_padded = self.preprocess_stereo(imgL, imgR)
        cv2.imwrite("data/images/processed_left.png", imgL_padded)
        cv2.imwrite("data/images/processed_right.png", imgR_padded)
        print("✅ Сохранены изображения после препроцессинга: processed_left.png, processed_right.png")

        if self.use_hailo:
            imgL_resized = np.ascontiguousarray(imgL_padded.astype(np.uint8)).reshape(1, 368, 1232, 3)
            imgR_resized = np.ascontiguousarray(imgR_padded.astype(np.uint8)).reshape(1, 368, 1232, 3)

            input_data = {"stereonet/input_layer1": imgL_resized, "stereonet/input_layer2": imgR_resized}

            with self.infer_vstreams as infer_pipeline:
                with self.configured_network.activate():
                    output_data = infer_pipeline.infer(input_data)
                    disparity = output_data.get("stereonet/conv53")

            if disparity is None or disparity.size == 0:
                raise ValueError("Ошибка: disparity пуст или не получен от модели!")

            disparity = np.squeeze(disparity)

            # Визуализация disparity перед ресайзом
            cv2.imshow("Disparity Map (Before Resize)",
                       (disparity - disparity.min()) / (disparity.max() - disparity.min()))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Масштабируем disparity обратно
            disparity = cv2.resize(disparity, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            disparity = self.stereo.compute(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

        focal_length = self.mtxL[0, 0]
        depth_map = (focal_length * self.baseline) / (disparity + 1e-6)

        # Генерируем визуализацию depth map
        depth_visual = cv2.applyColorMap(cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                                         cv2.COLORMAP_JET)

        print(f"Размер оригинального изображения: {imgL.shape}")
        print(f"Размер disparity map перед ресайзом: {disparity.shape}")
        print(f"Размер disparity map после ресайза: {depth_visual.shape}")

        # Накладываем depth map на оригинальное изображение
        depth_overlay = cv2.addWeighted(imgL, 0.5, depth_visual, 0.5, 0)
        cv2.imshow("Overlay Depth on Original", depth_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
