import cv2
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm
)


class LabelLoader:
    """Класс для загрузки классов объектов из текстового файла."""
    def __init__(self, label_path: str):
        self.label_path = label_path
        self.labels = self._load_labels()

    def _load_labels(self):
        """Загружает классы объектов из текстового файла построчно."""
        try:
            with open(self.label_path, "r", encoding="utf-8") as f:
                labels = {i: line.strip() for i, line in enumerate(f.readlines())}
            return labels
        except FileNotFoundError:
            print(f"[ERROR] Labels file '{self.label_path}' not found!")
            return {}

    def get_label(self, class_id: int) -> str:
        """Возвращает название класса по индексу."""
        return self.labels.get(class_id, f"Class {class_id}")


class InferenceImage:
    def __init__(self, image: np.ndarray, label_loader: LabelLoader):
        self.image = image
        self.label_loader = label_loader
        self.model_w = None
        self.model_h = None
        self.scale = None
        self.new_img_w = None
        self.new_img_h = None
        self.pasted_w = None
        self.pasted_h = None
        self.padded_image = None

    def set_model_input_size(self, model_w, model_h):
        self.model_w = model_w
        self.model_h = model_h

    def preprocess(self):
        img_h, img_w, _ = self.image.shape
        self.scale = min(self.model_w / img_w, self.model_h / img_h)
        self.new_img_w, self.new_img_h = int(img_w * self.scale), int(img_h * self.scale)
        image_resized = cv2.resize(self.image, (self.new_img_w, self.new_img_h))

        # Создаём новое изображение с отступами (padding)
        self.padded_image = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)
        self.pasted_w = (self.model_w - self.new_img_w) // 2
        self.pasted_h = (self.model_h - self.new_img_h) // 2
        self.padded_image[self.pasted_h:self.pasted_h + self.new_img_h, self.pasted_w:self.pasted_w+self.new_img_w, :] = image_resized
        return self.padded_image

    def postprocess(self, detection_results: dict):
        # Восстановление координат объектов в оригинальном изображении
        boxes = detection_results.get('detection_boxes', [])
        absolute_boxes = []
        for box in boxes:
            abs_coords = [
                int((coord * self.model_h - self.pasted_h) / self.scale) if i % 2 == 0
                else int((coord * self.model_w - self.pasted_w) / self.scale)
                for i, coord in enumerate(box)
            ]
            absolute_boxes.append(abs_coords)

        detection_results['absolute_boxes'] = absolute_boxes
        return detection_results

    def draw_boxes(self, results: dict):
        boxes = results.get('absolute_boxes', [])
        scores = results.get('detection_scores', [])
        classes = results.get('detection_classes', [])

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            class_id = classes[i] if i < len(classes) else "Unknown"
            class_name = self.label_loader.get_label(class_id)  # Получаем имя класса
            score = scores[i] if i < len(scores) else 0.0
            label = f'{class_name} ({score:.2f})'

            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return self.image


class HailoInference:
    def __init__(self, hef_path, label_path, output_type='FLOAT32'):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.label_loader = LabelLoader(label_path)  # Загружаем классы
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        return self.target.configure(self.hef, configure_params)[0]

    def _create_vstream_params(self, output_type):
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        return (
            InputVStreamParams.make_from_network_group(self.network_group, format_type=input_format_type),
            OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        )

    @staticmethod
    def extract_detections(input_data, conf_threshold=0.5):
        boxes, scores, classes = [], [], []
        num_detections = 0

        for i, detection in enumerate(input_data):
            for det in detection:
                bbox, score = det[:4], det[4]
                if score >= conf_threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1

        print(f'{classes=}')
        print(f'{scores=}')
        print(f'{num_detections=}')
        return {
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
            'num_detections': num_detections
        }

    def get_input_shape(self):
        return self.input_vstream_info[0].shape


class Processor:
    def __init__(self, inference: HailoInference, conf: float = 0.5):
        self._inference = inference
        self._conf = conf
        self.label_loader = inference.label_loader  # Доступ к labels

    def process(self, images: list):
        height, width, _ = self._inference.get_input_shape()
        processed_images = [InferenceImage(img, self.label_loader) for img in images]
        for img in processed_images:
            img.set_model_input_size(width, height)
            img.preprocess()

        raw_detections = self._inference.run(np.array([img.padded_image for img in processed_images]))
        results = []
        for img, detection in zip(processed_images, raw_detections):
            result = HailoInference.extract_detections(detection, self._conf)
            results.append(img.postprocess(result))

            drawed = img.draw_boxes(result)
            cv2.imshow("Camera", drawed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results
