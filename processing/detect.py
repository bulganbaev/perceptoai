import cv2
import numpy as np
import hailo_platform as hp


class YOLOv11sDetector:
    def __init__(self, use_hailo=True, hef_path="data/models/yolov11s.hef"):
        self.model_w, self.model_h = 640, 640  # Размер входа YOLOv11s
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

            print("✅ Hailo-8 успешно подключен. YOLOv11s загружен.")
            print("\n📌 Входные потоки:")
            for stream in self.input_vstream_infos:
                print(f" - {stream.name}")

            print("\n📌 Выходные потоки:")
            for stream in self.output_vstream_infos:
                print(f" - {stream.name}")

    def preprocess_with_padding(self, img_path):
        """Загружает и подготавливает изображение с паддингом"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Ошибка загрузки изображения: {img_path}")

        img_h, img_w, _ = img.shape
        scale = min(self.model_w / img_w, self.model_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        img_resized = cv2.resize(img, (new_w, new_h))
        img_padded = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)

        x_offset = (self.model_w - new_w) // 2
        y_offset = (self.model_h - new_h) // 2
        img_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = img_resized

        return np.ascontiguousarray(img_padded.astype(np.uint8)).reshape(1, 640, 640, 3)

    def infer(self, img_path):
        """Выполняет инференс и анализирует выходные данные"""
        img_preprocessed = self.preprocess_with_padding(img_path)
        input_data = {"yolov11s/input_layer1": img_preprocessed}

        with self.infer_vstreams as infer_pipeline:
            with self.configured_network.activate():
                output_data = infer_pipeline.infer(input_data)

        print("\n📌 Анализ выходных данных:")

        for key, value in output_data.items():
            if isinstance(value, list):  # Если это список, печатаем его длину
                print(f" - {key}: list of {len(value)} elements")
                if len(value) > 0 and isinstance(value[0], np.ndarray):
                    print(f"   └─ Первый элемент: shape={value[0].shape}, dtype={value[0].dtype}")
            elif isinstance(value, np.ndarray):
                print(f" - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f" - {key}: type={type(value)}")

        return output_data


if __name__ == "__main__":
    detector = YOLOv11sDetector(use_hailo=True)
    output_data = detector.infer("data/images/left/left_00.jpg")  # Укажи путь к изображению
