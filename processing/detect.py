import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)


class HailoInference:
    def __init__(self, hef_path, output_type='FLOAT32'):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        return self.target.configure(self.hef, configure_params)[0]

    def _create_vstream_params(self, output_type):
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()
        print("📌 Входные и выходные слои модели YOLOv11s:")
        for layer_info in input_vstream_info:
            print(f'➡ Вход: {layer_info.name} {layer_info.shape}')
        for layer_info in output_vstream_info:
            print(f'⬅ Выход: {layer_info.name} {layer_info.shape}')
        return input_vstream_info, output_vstream_info

    def run(self, input_data):
        input_dict = {self.input_vstream_info[0].name: input_data}
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                return infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]

    @staticmethod
    def extract_detections(input_data, conf_threshold=0.5):
        boxes, scores = [], []
        for detection in input_data:
            if detection.shape[0] > 0:
                for det in detection:
                    if det[4] > conf_threshold:
                        boxes.append(det[:4])  # x1, y1, x2, y2
                        scores.append(det[4])  # conf
        return {'boxes': np.array(boxes), 'scores': np.array(scores)}

    def release_device(self):
        self.target.release()


class ImageProcessor:
    def __init__(self, model_w, model_h):
        self.model_w, self.model_h = model_w, model_h

    def preprocess(self, image):
        """Масштабирует изображение с паддингами."""
        img_h, img_w, _ = image.shape
        scale = min(self.model_w / img_w, self.model_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        image_resized = cv2.resize(image, (new_w, new_h))

        padded_image = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)
        offset_w = (self.model_w - new_w) // 2
        offset_h = (self.model_h - new_h) // 2
        padded_image[offset_h:offset_h + new_h, offset_w:offset_w + new_w, :] = image_resized

        return padded_image, scale, offset_w, offset_h

    def restore_boxes(self, boxes, scale, offset_w, offset_h):
        """Конвертирует нормализованные координаты в пиксели."""
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * self.model_w - offset_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * self.model_h - offset_h) / scale
        return boxes.astype(int)


if __name__ == "__main__":
    model_path = "data/models/yolov11s.hef"
    image_path = "data/images/left/left_00.jpg"

    detector = HailoInference(model_path)
    model_w, model_h = detector.get_input_shape()[1:3]

    processor = ImageProcessor(model_w, model_h)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Ошибка загрузки изображения!")

    img_padded, scale, offset_w, offset_h = processor.preprocess(img)
    img_input = np.ascontiguousarray(img_padded.astype(np.uint8)).reshape(1, model_h, model_w, 3)

    print("📌 Запускаем YOLOv11s...")
    raw_output = detector.run(img_input)

    print("📌 Парсим результат...")
    results = detector.extract_detections(raw_output)

    if len(results["boxes"]) > 0:
        restored_boxes = processor.restore_boxes(results["boxes"], scale, offset_w, offset_h)
        print(f"✅ Найдено {len(restored_boxes)} объектов:")
        for i, box in enumerate(restored_boxes):
            print(f" - {i+1}: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}, score={results['scores'][i]:.2f}")
    else:
        print("❌ Объекты не найдены.")

    detector.release_device()
