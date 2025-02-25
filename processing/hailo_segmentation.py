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

        self.padded_image = np.zeros((self.model_w, self.model_h, 3), dtype=np.uint8)
        self.pasted_w = (self.model_w - self.new_img_w) // 2
        self.pasted_h = (self.model_h - self.new_img_h) // 2
        self.padded_image[self.pasted_h:self.pasted_h + self.new_img_h, self.pasted_w:self.pasted_w+self.new_img_w, :] = image_resized
        return self.padded_image

    def preprocessed(self):
        return self.padded_image

    def extract_segmentation_mask(self, model_outputs):
        output_layer = max(model_outputs.keys(), key=lambda k: model_outputs[k].shape[0])
        mask = model_outputs[output_layer]  # Выбираем слой с наибольшим разрешением
        mask = mask.astype(np.float32) / 255.0  # Нормализуем значения
        return mask

    def overlay_segmentation(self, mask):
        """
        Накладываем маску сегментации на изображение.
        """
        if mask is None or mask.size == 0:
            print("⚠️ Ошибка: Маска сегментации пуста!")
            return self.image  # Возвращаем оригинальное изображение без изменений

        mask_resized = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LINEAR)

        color_mask = np.zeros_like(self.image, dtype=np.uint8)
        color_mask[:, :, 2] = (mask_resized * 255).astype(np.uint8)  # Добавляем в синий канал

        return cv2.addWeighted(self.image, 0.7, color_mask, 0.3, 0)


class HailoSegmentation:
    def __init__(self, hef_path, output_type='UINT8'):
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
        input_format_type = self.hef.get_input_vstream_infos()[-1].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()
        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        return self.input_vstream_info[0].shape

    def run(self, input_data):
        input_dict = {self.input_vstream_info[0].name: input_data}
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)
        return output

    def release_device(self):
        self.target.release()

class Processor:
    def __init__(self, inference: HailoSegmentation):
        self._inference = inference

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

        yolo_output = self._inference.run(np.asarray(preprocessed_images))
        final_result = []

        for im in inf_images:
            mask = im.extract_segmentation_mask(yolo_output)
            segmented_image = im.overlay_segmentation(mask)
            cv2.imshow("Segmented Image", segmented_image)
            cv2.waitKey(1)
            final_result.append(mask)

        elapsed_time = time.time() - start_time
        print(f"[INFO] Total elapsed time: {elapsed_time:.3f} seconds")
        return final_result
