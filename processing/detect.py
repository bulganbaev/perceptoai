import cv2
import numpy as np
import os
import time
import hailo_platform as hp

class DepthEstimator:
    def __init__(self, calib_path="data/calibration/calibration_data.npz", use_hailo=True,
                 hef_path="data/models/yolov11s.hef"):
        # Загружаем параметры калибровки
        calib_data = np.load(calib_path)
        self.mtxL, self.distL = calib_data["mtxL"], calib_data["distL"]
        self.mtxR, self.distR = calib_data["mtxR"], calib_data["distR"]
        self.R, self.T = calib_data["R"], calib_data["T"]

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.mtxL, self.distL, self.mtxR, self.distR, (1920, 1080), self.R, self.T, alpha=1)

        self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(self.mtxL, self.distL, self.R1, self.P1, (1920, 1080), cv2.CV_16SC2)
        self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.mtxR, self.distR, self.R2, self.P2, (1920, 1080), cv2.CV_16SC2)

        self.model_w, self.model_h = 640, 640
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
                                                                                         format_type=hp.FormatType.FLOAT32)

            self.infer_vstreams = hp.InferVStreams(self.configured_network, self.input_vstreams_params,
                                                   self.output_vstreams_params)

            print("✅ Hailo-8 успешно подключен. YOLO11s загружен.")

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

    import numpy as np

    def detect_objects(self, img):
        """
        Запускает детекцию объектов через YOLO11s на Hailo и парсит результат.
        """
        img_resized = cv2.resize(img, (640, 640))  # Размер входа для YOLO
        img_resized = np.ascontiguousarray(img_resized.astype(np.uint8)).reshape(1, 640, 640, 3)

        input_data = {"yolov11s/input_layer1": img_resized}

        with self.infer_vstreams as infer_pipeline:
            with self.configured_network.activate():
                output_data = infer_pipeline.infer(input_data)

        detections_raw = output_data.get("yolov11s/yolov8_nms_postprocess")

        if not detections_raw:
            print("❌ Объекты не найдены!")
            return []

        # ✅ Фильтруем только массивы `numpy`, игнорируем пустые списки
        filtered_detections = [np.array(det) for det in detections_raw if isinstance(det, np.ndarray) and len(det) > 0]

        if len(filtered_detections) == 0:
            print("❌ Объекты не найдены после фильтрации!")
            return []

        # ✅ Объединяем все найденные объекты в один массив
        detections = np.vstack(filtered_detections)

        print(f"✅ Найдено объектов: {detections.shape[0]}")

        return detections  # [x1, y1, x2, y2, score, class]

    def compute_detection(self, imgL_path, imgR_path):
        start_time = time.time()

        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)

        if imgL is None or imgR is None:
            raise ValueError("Ошибка загрузки изображений! Проверьте пути.")

        # ✅ Делаем ректификацию
        imgL_rect = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
        imgR_rect = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

        # ✅ Применяем масштабирование и центрирование
        imgL_padded, imgR_padded = self.preprocess_stereo(imgL_rect, imgR_rect)

        # ✅ Сохраняем изображения после препроцессинга для проверки
        cv2.imwrite("data/images/processed_left.png", imgL_padded)
        cv2.imwrite("data/images/processed_right.png", imgR_padded)
        print("✅ Сохранены нормализованные изображения: processed_left.png, processed_right.png")

        # ✅ Запускаем YOLO на обеих камерах
        detections_left = self.detect_objects(imgL_padded)
        detections_right = self.detect_objects(imgR_padded)

        # ✅ Рисуем bounding box'ы
        for det in detections_left:
            x1, y1, x2, y2, score, cls = det
            cv2.rectangle(imgL_padded, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(imgL_padded, f"ID:{int(cls)} {score:.2f}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for det in detections_right:
            x1, y1, x2, y2, score, cls = det
            cv2.rectangle(imgR_padded, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(imgR_padded, f"ID:{int(cls)} {score:.2f}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Сохраняем изображения с детекцией
        cv2.imwrite("data/images/detected_left.png", imgL_padded)
        cv2.imwrite("data/images/detected_right.png", imgR_padded)
        print("✅ Сохранены результаты детекции: detected_left.png, detected_right.png")

        elapsed_time = time.time() - start_time
        print(f"⏱ Время выполнения: {elapsed_time:.4f} сек")


if __name__ == "__main__":
    detector = DepthEstimator(use_hailo=True)
    detector.compute_detection("data/images/left/left_00.jpg", "data/images/right/right_00.jpg")
