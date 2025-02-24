#!/usr/bin/env python3
"""
Модуль для выполнения детекции с использованием HailoInference.
Содержит классы: InferenceImage, HailoInference, Processor.
"""

import cv2
import numpy as np
import time
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, HailoSchedulingAlgorithm)


class InferenceImage:
    def __init__(self, image: np.ndarray):
        """
        Инициализация объекта для обработки изображения.
        """
        self.image = image  # Исходное изображение
        self.img_h, self.img_w, _ = self.image.shape  # Высота и ширина исходного изображения
        # Размеры входного слоя модели (будут заданы)
        self.model_h = None  # Высота модели (pixels)
        self.model_w = None  # Ширина модели (pixels)
        self.scale = None  # Коэффициент масштабирования
        self.new_img_h = None  # Новая высота после масштабирования
        self.new_img_w = None  # Новая ширина после масштабирования
        self.pasted_h = None  # Отступ сверху
        self.pasted_w = None  # Отступ слева
        self.padded_image = None  # Изображение с добавленным паддингом

    def set_model_input_size(self, model_h: int, model_w: int):
        """
        Задает размеры входного слоя модели.

        Параметры:
            model_h (int): Высота входного слоя модели.
            model_w (int): Ширина входного слоя модели.
        """
        self.model_h = model_h
        self.model_w = model_w

    def preprocess(self):
        """
        Масштабирует изображение до размеров модели с сохранением пропорций,
        добавляет паддинг для соответствия входному размеру модели.

        Возвращает:
            np.ndarray: Подготовленное изображение.
        """
        # Вычисляем коэффициент масштабирования для подгона изображения под модель
        self.scale = min(self.model_w / self.img_w, self.model_h / self.img_h)
        self.new_img_w = int(self.img_w * self.scale)
        self.new_img_h = int(self.img_h * self.scale)

        # Изменяем размер исходного изображения
        image_resized = cv2.resize(self.image, (self.new_img_w, self.new_img_h))

        # Создаем пустое изображение (паддинг) с формой (model_h, model_w, 3)
        self.padded_image = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)

        # Вычисляем отступы для центрирования изображения
        self.pasted_w = (self.model_w - self.new_img_w) // 2
        self.pasted_h = (self.model_h - self.new_img_h) // 2

        # Вставляем масштабированное изображение в центр пустого изображения
        self.padded_image[self.pasted_h:self.pasted_h + self.new_img_h,
        self.pasted_w:self.pasted_w + self.new_img_w, :] = image_resized

        return self.padded_image

    def postprocess(self, detection_results: dict):
        """
        Восстанавливает координаты обнаруженных объектов в исходное изображение.

        Параметры:
            detection_results (dict): Результаты детекции с нормализованными координатами.

        Возвращает:
            dict: Результаты детекции с абсолютными координатами.
        """
        boxes = detection_results.get('detection_boxes', [])
        absolute_boxes = []
        for box in boxes:
            abs_coords = []
            # Предполагается, что порядок координат: [y1, x1, y2, x2]
            for i, coord in enumerate(box):
                if i % 2 == 0:
                    # Координата по высоте (y)
                    abs_coord = coord * self.model_h - self.pasted_h
                else:
                    # Координата по ширине (x)
                    abs_coord = coord * self.model_w - self.pasted_w
                # Восстанавливаем исходный масштаб
                abs_coord /= self.scale
                abs_coords.append(int(abs_coord))
            absolute_boxes.append(abs_coords)
        detection_results.update({'absolute_boxes': absolute_boxes})
        return detection_results

    def postprocess_mask(self, detection_results: dict):
        """
        Восстанавливает сегментационные маски к оригинальному размеру изображения.

        Параметры:
            detection_results (dict): Результаты детекции, содержащие ключ 'segmentation_masks'.

        Возвращает:
            dict: Результаты с восстановленными масками в ключе 'absolute_masks'.
        """
        masks = detection_results.get('segmentation_masks')
        if masks is None:
            return detection_results

        restored_masks = []
        for mask in masks:
            # Изменяем размер маски к размеру входного слоя модели
            mask_resized = cv2.resize(mask, (self.model_w, self.model_h), interpolation=cv2.INTER_NEAREST)

            # Обмен осей, если требуется (в некоторых моделях X и Y могут быть перепутаны)
            mask_transposed = mask_resized.T

            # Обрезаем паддинг, чтобы оставить область с изображением
            mask_cropped = mask_transposed[self.pasted_h:self.pasted_h + self.new_img_h,
                           self.pasted_w:self.pasted_w + self.new_img_w]

            # Восстанавливаем исходный размер изображения
            restored_mask = cv2.resize(mask_cropped, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            restored_masks.append(restored_mask)

        detection_results.update({'absolute_masks': np.array(restored_masks, dtype=np.uint8)})
        return detection_results

    def draw_boxes(self, results: dict):
        """
        Рисует прямоугольники обнаруженных объектов на изображении.

        Параметры:
            results (dict): Результаты детекции, содержащие ключи 'absolute_boxes', 'detection_scores', 'detection_classes'.

        Возвращает:
            np.ndarray: Изображение с нарисованными прямоугольниками.
        """
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
    def __init__(self, hef_path: str, output_type: str = 'FLOAT32'):
        """
        Инициализация HailoInference с использованием HEF модели.

        Параметры:
            hef_path (str): Путь к HEF файлу модели.
            output_type (str): Тип выходных данных модели.
        """
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        """
        Конфигурирует устройство Hailo и возвращает группу сетей.
        """
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, output_type: str):
        """
        Создает параметры для входного и выходного потоков.

        Параметры:
            output_type (str): Тип формата для выходного потока.

        Возвращает:
            tuple: Параметры входного и выходного потоков.
        """
        input_format_type = self.hef.get_input_vstream_infos()[-1].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group,
                                                                           format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group,
                                                                             format_type=getattr(FormatType,
                                                                                                 output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        """
        Получает и выводит информацию о входных и выходных потоках.

        Возвращает:
            tuple: Списки информации о входных и выходных потоках.
        """
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            print(f'Input layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        for layer_info in output_vstream_info:
            print(f'Output layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')

        return input_vstream_info, output_vstream_info

    @staticmethod
    def extract_detections(input_data, conf_threshold: float = 0.5) -> dict:
        """
        Извлекает детекции из выходных данных модели.

        Параметры:
            input_data (list): Сырые данные детекции.
            conf_threshold (float): Порог уверенности для фильтрации детекций.

        Возвращает:
            dict: Отфильтрованные детекции.
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

    @staticmethod
    def extract_segmentations(input_data: np.ndarray, conf_threshold: float = 0.5) -> dict:
        """
        Извлекает сегментационные данные из выходных данных модели YOLOv8.

        Параметры:
            input_data (np.ndarray): Выходные данные сегментации модели, форма (H, W, num_classes).
            conf_threshold (float): Порог уверенности для фильтрации сегментаций.

        Возвращает:
            dict: Извлеченные сегментации, включая маски, ограничивающие рамки и оценки.
        """
        height, width, num_classes = input_data.shape
        masks, bounding_boxes, scores, classes = [], [], [], []

        for class_id in range(num_classes):
            class_map = input_data[:, :, class_id]
            binary_mask = (class_map > conf_threshold).astype(np.uint8)
            if np.sum(binary_mask) == 0:
                continue
            # Поиск контуров для текущего класса
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                masks.append(binary_mask)
                bounding_boxes.append([x, y, x + w, y + h])
                scores.append(np.max(class_map))
                classes.append(class_id)

        return {
            'segmentation_masks': masks,
            'bounding_boxes': bounding_boxes,
            'detection_scores': scores,
            'detection_classes': classes,
            'num_segmentations': len(masks)
        }

    def get_input_shape(self):
        """
        Получает форму входного слоя модели.

        Возвращает:
            tuple: Форма входного слоя (высота, ширина, каналы).
        """
        # Предполагается, что модель имеет один вход
        return self.input_vstream_info[0].shape

    def run(self, input_data):
        """
        Запускает инференс на устройстве Hailo-8.

        Параметры:
            input_data: Данные для инференса (np.ndarray, dict, list, tuple).

        Возвращает:
            Выход инференса.
        """
        input_dict = self._prepare_input_data(input_data)
        with InferVStreams(self.network_group, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]
        return output

    def _prepare_input_data(self, input_data):
        """
        Подготавливает данные для инференса.

        Параметры:
            input_data: Данные для инференса.

        Возвращает:
            dict: Подготовленные данные.
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
        Освобождает устройство Hailo.
        """
        self.target.release()


class Processor:
    def __init__(self, inference: HailoInference, conf: float = 0.5):
        """
        Инициализация процессора для обработки изображений с инференсом.

        Параметры:
            inference (HailoInference): Экземпляр класса HailoInference.
            conf (float): Порог уверенности для детекции.
        """
        self._inference = inference
        self._conf = conf

    def process(self, images: list) -> list:
        """
        Обрабатывает список изображений, выполняет инференс и постобработку результатов.

        Параметры:
            images (list): Список изображений (np.ndarray).

        Возвращает:
            list: Список результатов инференса с постобработкой.
        """
        start_time = time.time()
        inf_images = []
        # Получаем форму входного слоя модели: (model_h, model_w, channels)
        model_h, model_w, _ = self._inference.get_input_shape()
        preprocessed_images = []
        for im in images:
            inf_img = InferenceImage(im)
            # Передаем правильный порядок: сначала высота, затем ширина
            inf_img.set_model_input_size(model_h, model_w)
            preprocessed = inf_img.preprocess()
            preprocessed_images.append(preprocessed)
            inf_images.append(inf_img)

        # Стекуем изображения для инференса
        raw_detect_data = self._inference.run(np.stack(preprocessed_images))
        final_result = []
        for det, inf_img in zip(raw_detect_data, inf_images):
            # Извлекаем сегментации из результатов инференса
            result = HailoInference.extract_segmentations(det, self._conf)
            # Постобработка масок для восстановления исходного размера изображения
            final_result.append(inf_img.postprocess_mask(result))

        elapsed_time = time.time() - start_time
        print(f"[INFO] Total elapsed time: {elapsed_time:.3f} seconds")
        return final_result
