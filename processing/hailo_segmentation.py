import cv2
import numpy as np
import time
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)


class InferenceImageSegmentation:
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
        """ Изменение размера входного изображения под размер модели """
        img_h, img_w, _ = self.image.shape
        self.scale = min(self.model_w / img_w, self.model_h / img_h)
        self.new_img_w, self.new_img_h = int(img_w * self.scale), int(img_h * self.scale)
        image_resized = cv2.resize(self.image, (self.new_img_w, self.new_img_h))

        # Создание нового изображения с паддингом
        self.padded_image = np.zeros((self.model_h, self.model_w, 3), dtype=np.uint8)
        self.pasted_w = (self.model_w - self.new_img_w) // 2
        self.pasted_h = (self.model_h - self.new_img_h) // 2
        self.padded_image[self.pasted_h:self.pasted_h + self.new_img_h, self.pasted_w:self.pasted_w + self.new_img_w,
        :] = image_resized
        return self.padded_image

    def postprocess(self, segmentation_output: np.ndarray):
        """ Восстановление размеров сегментированной маски под оригинальное изображение """
        if segmentation_output is None or segmentation_output.size == 0:
            raise ValueError("Ошибка: Пустая сегментированная карта. Проверьте входные данные модели.")

        # Обрезка паддинга и восстановление масштаба
        segmentation_output = segmentation_output[self.pasted_h:self.pasted_h + self.new_img_h,
                              self.pasted_w:self.pasted_w + self.new_img_w]
        segmentation_output = cv2.resize(segmentation_output, (self.image.shape[1], self.image.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
        return segmentation_output

    def overlay_segmentation(self, segmentation_map: np.ndarray, alpha=0.5):
        """ Наложение сегментационной маски на оригинальное изображение """
        if segmentation_map is None or segmentation_map.size == 0:
            raise ValueError("Ошибка: Пустая сегментационная карта.")

        colors = np.random.randint(0, 255, size=(np.max(segmentation_map) + 1, 3), dtype=np.uint8)
        color_mask = colors[segmentation_map]
        blended = cv2.addWeighted(self.image, 1 - alpha, color_mask, alpha, 0)
        return blended


class HailoSegmentation:
    def __init__(self, hef_path, output_type='FLOAT32'):
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, output_type):
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group,
                                                                           format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group,
                                                                             format_type=getattr(FormatType,
                                                                                                 output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            print(f'Input layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        for layer_info in output_vstream_info:
            print(f'Output layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')

        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        return self.input_vstream_info[0].shape

    def run(self, input_data):
        input_dict = self._prepare_input_data(input_data)

        with InferVStreams(self.network_group, self.input_vstreams_params,
                           self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]

        return output

    def _prepare_input_data(self, input_data):
        if isinstance(input_data, dict):
            return input_data
        elif isinstance(input_data, (list, tuple)):
            return {self.input_vstream_info[0].name: np.array(input_data)}
        else:
            return {self.input_vstream_info[0].name: np.expand_dims(input_data, axis=0)}

    def release_device(self):
        self.target.release()


class ProcessorSegmentation:
    def __init__(self, inference: HailoSegmentation):
        self._inference = inference

    def process(self, images: list):
        start_time = time.time()
        inf_images = []
        height, width, _ = self._inference.get_input_shape()
        preprocessed_images = []

        for im in images:
            inf_img = InferenceImageSegmentation(im)
            inf_img.set_model_input_size(width, height)
            preprocessed_images.append(inf_img.preprocess())
            inf_images.append(inf_img)

        raw_segmentation_data = self._inference.run(np.asarray(preprocessed_images))
        final_result = []

        for seg_map, im in zip(raw_segmentation_data, inf_images):
            seg_map = np.argmax(seg_map, axis=0)  # Выбираем класс с максимальной вероятностью
            seg_map_resized = im.postprocess(seg_map)
            overlay = im.overlay_segmentation(seg_map_resized)
            final_result.append(overlay)

        elapsed_time = time.time() - start_time
        print(f"[INFO] Total elapsed time: {elapsed_time:.3f} seconds")
        return final_result
