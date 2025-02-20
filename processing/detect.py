import cv2
import numpy as np
import hailo_platform as hp

class ObjectDetector:
    def __init__(self, model_path="data/models/yolov11s.hef", use_hailo=True):
        self.model_w, self.model_h = 640, 640  # Размер входа YOLO

        self.use_hailo = use_hailo
        if use_hailo:
            self.vdevice = hp.VDevice()
            self.hef = hp.HEF(model_path)
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

    def preprocess_image(self, image):
        """Подготовка изображения: ресайз с паддингами."""
        img_h, img_w, _ = image.shape
        scale = min(self.model_w / img_w, self.model_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))

        padded_image = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)
        offset_w = (self.model_w - new_w) // 2
        offset_h = (self.model_h - new_h) // 2
        padded_image[offset_h:offset_h + new_h, offset_w:offset_w + new_w, :] = image_resized

        return padded_image

    def detect_objects(self, img):
        """Запускает YOLOv11s на Hailo и парсит результат."""
        img_padded = self.preprocess_image(img)
        img_input = np.ascontiguousarray(img_padded.astype(np.uint8)).reshape(1, 640, 640, 3)

        input_data = {"yolov11s/input_layer1": img_input}

        with self.infer_vstreams as infer_pipeline:
            with self.configured_network.activate():
                output_data = infer_pipeline.infer(input_data)

        return output_data["yolov11s/yolov8_nms_postprocess"]

    def process_yolo_output(self, yolo_output, img_w=640, img_h=640, conf_thresh=0.5):
        """Обрабатывает выход YOLOv11s и возвращает боксы в пикселях."""
        detections = yolo_output[0]  # YOLO возвращает список из 1 элемента

        if isinstance(detections, np.ndarray) and detections.shape[-1] == 5:
            print("🎯 Данные в формате [x1, y1, x2, y2, score]")

            # Фильтруем боксы по confidence
            filtered_boxes = detections[detections[:, 4] > conf_thresh]

            # Конвертируем из нормализованных [0,1] в пиксели
            filtered_boxes[:, [0, 2]] *= img_w  # x1, x2 → пиксели
            filtered_boxes[:, [1, 3]] *= img_h  # y1, y2 → пиксели

            print(f"✅ Найдено {filtered_boxes.shape[0]} объектов с conf > {conf_thresh}")
            print(filtered_boxes)

            return filtered_boxes
        else:
            print("❌ Неизвестный формат данных!")
            return []

    def compute_detection(self, img_path):
        """Запускает полный пайплайн детекции."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Ошибка загрузки изображения!")

        print("📌 Запускаем детекцию объектов...")
        raw_output = self.detect_objects(img)
        processed_boxes = self.process_yolo_output(raw_output)

        return processed_boxes


if __name__ == "__main__":
    detector = ObjectDetector()
    detections = detector.compute_detection("data/images/left/left_00.jpg")
