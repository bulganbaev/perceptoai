import cv2
import numpy as np
import threading
import time
import logging
from picamera2 import Picamera2
from libcamera import controls

# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("stereo_camera.log"),
        logging.StreamHandler()
    ]
)


class CameraDriver:
    """
    Драйвер для камеры Arducam 16MP IMX519 с фиксированной экспозицией и усилением.
    """

    def __init__(self, camera_id=0, width=1920, height=1080, flip_horizontal = False, flip_vertical = False):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.exposure_time = 4000  # Увеличенная экспозиция (4 мс)
        self.analogue_gain = 3.0  # Сильное усиление
        self.digital_gain = 1.5  # Цифровое усиление для компенсации
        self.colour_gains = (1.5, 1.5)  # Усиленный баланс белого
        self.contrast = 1.5  # Контраст выше стандартного
        self.saturation = 1.3  # Немного увеличенная насыщенность
        self.lens_position = 2.0  # Фиксированный фокус (1.5 - 3.0)

        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal

        try:
            self.picam = Picamera2(camera_id)
            controls_list = self.picam.camera_controls
            control_params = {}

            # control_params = {
            #     "AfMode": controls.AfModeEnum.Manual,  # Отключаем автофокус
            #     "LensPosition": self.lens_position,
            #     "AeEnable": 0,  # Отключаем автоэкспозицию
            #     "ExposureTime": self.exposure_time,
            #     "AnalogueGain": self.analogue_gain,
            #     "AeMeteringMode": controls.AeMeteringModeEnum.Spot,  # Улучшенная работа в сложном освещении
            #     "AeExposureMode": controls.AeExposureModeEnum.Short,  # Короткая выдержка
            #     "ColourGains": self.colour_gains,  # Новое: коррекция баланса белого
            #     "Contrast": self.contrast,  # Новое: повышение контраста
            #     "Saturation": self.saturation  # Новое: насыщенность выше стандартного
            # }

            config = self.picam.create_still_configuration(
                main={'size': (self.width, self.height)},
                controls=control_params
            )
            self.picam.configure(config)

            logging.info(f"Камера {self.camera_id} успешно инициализирована.")

        except Exception as e:
            logging.error(f"Ошибка при инициализации камеры {camera_id}: {e}")
            self.picam = None

    def start_camera(self):
        """Запускает поток захвата изображения"""
        if self.running or self.picam is None:
            return
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        logging.info(f"Камера {self.camera_id} запущена.")

    def _capture_loop(self):
        """Основной поток захвата изображений"""
        try:
            self.picam.start()
            while self.running:
                frame = self.picam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Применяем инверсию кадра
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)  # Отражение по горизонтали
                if self.flip_vertical:
                    frame = cv2.flip(frame, 0)
                self.frame = frame
        except Exception as e:
            logging.error(f"Ошибка в потоке камеры {self.camera_id}: {e}")
        finally:
            self.picam.stop()

    def get_frame(self):
        """Возвращает последний кадр"""
        return self.frame

    def stop_camera(self):
        """Останавливает поток"""
        self.running = False
        if self.picam:
            self.picam.close()
        logging.info(f"Камера {self.camera_id} остановлена.")


class StereoCameraSystem:
    """Система стереокамер с синхронизацией параметров."""

    def __init__(self, camera0_id=0, camera1_id=1):
        self.cam0 = CameraDriver(camera_id=camera0_id, flip_vertical=True, flip_horizontal=True)
        self.cam1 = CameraDriver(camera_id=camera1_id, flip_vertical=True, flip_horizontal=True)

        logging.info("Система стереокамер инициализирована.")

    def start(self):
        """Запускает обе камеры"""
        self.cam0.start_camera()
        self.cam1.start_camera()
        logging.info("Обе камеры запущены.")

    def get_synchronized_frames(self):
        """Возвращает последние кадры с обеих камер"""
        return self.cam0.get_frame(), self.cam1.get_frame()

    def stop(self):
        """Останавливает обе камеры"""
        self.cam0.stop_camera()
        self.cam1.stop_camera()
        logging.info("Система стереокамер остановлена.")
