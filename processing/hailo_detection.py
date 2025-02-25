import cv2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm)
import numpy as np
import time



class InferenceImage:
    def __init__(self, image: np.ndarray):
        self.image = image
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

        # Create a new padded image
        self.padded_image = np.zeros((self.model_w, self.model_h, 3), dtype=np.uint8)
        self.pasted_w = (self.model_w - self.new_img_w) // 2
        self.pasted_h = (self.model_h - self.new_img_h) // 2
        self.padded_image[self.pasted_h:self.pasted_h + self.new_img_h, self.pasted_w:self.pasted_w+self.new_img_w, :] = image_resized
        return self.padded_image

    def preprocessed(self):
        return self.padded_image

    def postprocess(self, detection_results: dict):
        # as of now just restore the original coordinates in the image
        boxes = detection_results.get('detection_boxes')
        absolute_boxes = []
        for box in boxes:
            abs_coords = []
            for i, coord in enumerate(box):
                if i % 2 == 0:
                    # height (y) is first coming
                    abs_coord = coord * self.model_h
                    abs_coord -= self.pasted_h
                else:
                    # getting real coordinates first
                    abs_coord = coord * self.model_w
                    # get a coordinate without padding
                    abs_coord -= self.pasted_w
                # restore original coordinates
                abs_coord /= self.scale
                abs_coords.append(int(abs_coord))
            absolute_boxes.append(abs_coords)

        detection_results.update({'absolute_boxes': absolute_boxes})
        return detection_results

    def draw_boxes(self, results: dict):
        boxes = results.get('absolute_boxes', [])
        scores = results.get('detection_scores', [])
        classes = results.get('detection_classes', [])

        for i, (y1, x1, y2, x2) in enumerate(boxes):
            class_id = classes[i] if i < len(classes) else "Unknown"
            score = scores[i] if i < len(scores) else 0.0
            label = f'{class_id} ({score:.2f})'

            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return self.image


class HailoInference:
    def __init__(self, hef_path,  output_type='FLOAT32'):
        """
        Initialize the HailoInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
        """
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        """
        Configure the Hailo device and get the network group.

        Returns:
            NetworkGroup: Configured network group.
        """
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, output_type):
        """
        Create input and output stream parameters.

        Args:
            output_type (str): Format type of the output stream.

        Returns:
            tuple: Input and output stream parameters.
        """
        input_format_type = self.hef.get_input_vstream_infos()[-1].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        """
        Get and print information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            print(f'Input layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        for layer_info in output_vstream_info:
            print(f'Output layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')

        return input_vstream_info, output_vstream_info

    @staticmethod
    def extract_detections(input_data, conf_threshold: float = 0.5):
        """
        Extract detections from the input data.

        Args:
            input_data (list): Raw detections from the model.
            threshold (float): Score threshold for filtering detections.

        Returns:
            dict: Filtered detection results.
        """
        boxes, scores, classes = [], [], []
        num_detections = 0

        for i, detection in enumerate(input_data):
            if len(detection) == 0:
                continue

            for det in detection:
                bbox, score = det[:4], det[4]

                if score >= conf_threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1

        return {
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
            'num_detections': num_detections
        }

    def get_input_shape(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info[0].shape  # Assumes that the model has one input

    def run(self, input_data):
        """
        Run inference on Hailo-8 device.

        Args:
            input_data (np.ndarray, dict, list, tuple): Input data for inference.

        Returns:
            np.ndarray: Inference output.
        """
        input_dict = self._prepare_input_data(input_data)

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]

        return output

    def _prepare_input_data(self, input_data):
        """
        Prepare input data for inference.

        Args:
            input_data (np.ndarray, dict, list, tuple): Input data for inference.

        Returns:
            dict: Prepared input data.
        """
        input_dict = {}
        if isinstance(input_data, dict):
            return input_data
        elif isinstance(input_data, (list, tuple)):
            for layer_info in self.input_vstream_info:
                input_dict[layer_info.name] = input_data
        else:
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)
            input_dict[self.input_vstream_info[0].name] = input_data

        return input_dict

    def release_device(self):
        """
        Release the Hailo device.
        """
        self.target.release()



class HailoSegmentation:
    def __init__(self, inference_model):
        """
        Класс для работы с моделью сегментации на Hailo.

        Args:
            inference_model (HailoInference): Экземпляр класса HailoInference.
        """
        self.inference = inference_model

    def run_inference(self, image):
        """
        Запускает инференс на одном изображении и выводит сырые данные.

        Args:
            image (np.ndarray): Изображение для обработки (формат RGB).

        Returns:
            dict: Сырые выходные данные модели.
        """
        start_time = time.time()

        # Подготовка изображения
        height, width, _ = self.inference.get_input_shape()
        input_image = self.preprocess(image, width, height)

        # Запуск инференса
        raw_outputs = self.inference.run(np.expand_dims(input_image, axis=0))

        elapsed_time = time.time() - start_time
        print(f"[INFO] Инференс выполнен за {elapsed_time:.3f} секунд")

        # Выводим информацию о выходных тензорах
        self.print_output_info(raw_outputs)

        return raw_outputs

    def preprocess(self, image, model_w, model_h):
        """
        Подготавливает изображение для подачи в модель.

        Args:
            image (np.ndarray): Исходное изображение.
            model_w (int): Ширина входного слоя модели.
            model_h (int): Высота входного слоя модели.

        Returns:
            np.ndarray: Подготовленное изображение.
        """
        image_resized = cv2.resize(image, (model_w, model_h))
        return image_resized

    @staticmethod
    def print_output_info(output_data):
        """
        Выводит информацию о выходных тензорах.

        Args:
            output_data (dict): Сырые выходные данные модели.
        """
        print("\n=== [ Выходные данные модели ] ===")
        for key, value in output_data.items():
            print(f"[INFO] Тензор: {key}")
            print(f"  - Форма: {value.shape}")
            print(f"  - Тип данных: {value.dtype}")
            print(f"  - Мин: {np.min(value)}, Макс: {np.max(value)}")
            print(f"  - Среднее значение: {np.mean(value)}\n")





class Processor:
    def __init__(self, inference: HailoInference, conf: float = 0.5):
        self._inference = inference
        self._conf = conf

    def process(self, images: list):
        start_time = time.time()
        inf_images = []
        height, width, _ = self._inference.get_input_shape()
        preprocessed_images = []
        for im in images:
            inf_img = InferenceImage(im)
            inf_img.set_model_input_size(width, height)
            preprocessed_images.append(inf_img.preprocess())
            inf_images.append(inf_img)
        raw_detect_data = self._inference.run(np.asarray(preprocessed_images))
        final_result = []
        for det, im in zip(raw_detect_data, inf_images):
            result = HailoInference.extract_detections(det, self._conf)
            final_result.append(im.postprocess(result))

            # drawed = im.draw_boxes(result)
            # cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # Делаем окно изменяемым
            # cv2.resizeWindow("Camera", 960, 540)
            # cv2.imshow("Camera", drawed)
        elapsed_time = time.time() - start_time  # Вычисляем общее время выполнения
        print(f"[INFO] Total elapsed time: {elapsed_time:.3f} seconds")
        return final_result
