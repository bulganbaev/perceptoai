import libcamera
import cv2
import numpy as np
import threading
import atexit
import time
import logging
from picamera2 import Picamera2

# Настройки логирования
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("stereo_camera.log"),
        logging.StreamHandler()
    ]
)


class CameraDriver:
    """
    Драйвер для камеры OV5647 с синхронизацией экспозиции и фокуса в фоне.
    """

    def __init__(self, camera_id=0, width=1920, height=1080, autofocus=True):
        self.lens_position = None
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.autofocus = autofocus
        self.auto_adjust = False
        self.last_adjust_time = time.time()
        self.exposure_time = 1000
        self.analogue_gain = 1.0
        self.colour_gains = (1.0, 1.0)
        self.contrast = 1.0
        self.saturation = 1.0
        self.sharpness = 1.0
        self.exposure_lock = threading.Lock()
        self.update_needed = threading.Event()

        try:
            self.picam = Picamera2(camera_id)
            controls = self.picam.camera_controls
            control_params = {}

            if "AfMode" in controls:
                control_params["AfMode"] = 2 if autofocus else 0
                control_params["AfSpeed"] = 1

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
        """Запускает потоки захвата изображения и автоэкспозиции"""
        if self.running or self.picam is None:
            return
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

        if self.auto_adjust:
            threading.Thread(target=self._exposure_adjust_loop, daemon=True).start()

        self.update_thread = threading.Thread(target=self._apply_settings_loop, daemon=True)
        self.update_thread.start()

        logging.info(f"Камера {self.camera_id} запущена.")

    def _capture_loop(self):
        """Основной поток захвата изображений"""
        try:
            self.picam.start()
            while self.running:
                frame = self.picam.capture_array()
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Ошибка в потоке камеры {self.camera_id}: {e}")
        finally:
            self.picam.stop()

    def _exposure_adjust_loop(self):
        """Фоновый поток автоэкспозиции"""
        while self.running:
            time.sleep(1)
            self.adjust_exposure()

    def adjust_exposure(self):
        """Автоматически корректирует экспозицию и фокус камеры."""
        if self.frame is None:
            return

        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)

        metadata = self.picam.capture_metadata()
        with self.exposure_lock:
            self.exposure_time = metadata.get("ExposureTime", 1000)
            self.analogue_gain = metadata.get("AnalogueGain", 1.0)
            self.colour_gains = metadata.get("ColourGains", (1.0, 1.0))
            self.contrast = metadata.get("Contrast", 1.0)
            self.saturation = metadata.get("Saturation", 1.0)
            self.sharpness = metadata.get("Sharpness", 1.0)

            if "LensPosition" in metadata:
                self.lens_position = metadata["LensPosition"]

            if avg_brightness < 50:
                self.exposure_time = min(self.exposure_time * 1.5, 600000)
                self.analogue_gain = min(self.analogue_gain * 1.2, 4)
            elif avg_brightness > 180:
                self.exposure_time = max(self.exposure_time * 0.7, 1000)
                self.analogue_gain = max(self.analogue_gain * 0.8, 1)

            self.update_needed.set()
        logging.warning(f'{avg_brightness=}')
        logging.warning(f'{self.exposure_time=}')
        logging.info(
            f"[Камера {self.camera_id}] Коррекция экспозиции: "
            f"ExposureTime={self.exposure_time}, "
            f"AnalogueGain={self.analogue_gain}, "
            f"LensPosition={self.lens_position}"
        )

    def _apply_settings_loop(self):
        """Фоновый поток обновления параметров камеры"""
        while self.running:
            self.update_needed.wait()
            self.update_needed.clear()
            self._apply_pending_settings()

    def _apply_pending_settings(self):
        """Применяет обновленные настройки камеры"""
        with self.exposure_lock:
            controls = {
                "AeEnable": 0,
                "ExposureTime": int(self.exposure_time),
                "AnalogueGain": self.analogue_gain,
                "ColourGains": self.colour_gains,
                "Contrast": self.contrast,
                "Saturation": self.saturation,
                "Sharpness": self.sharpness,
            }

            if "LensPosition" in self.picam.camera_controls and self.lens_position is not None:
                controls["LensPosition"] = self.lens_position

            self.picam.set_controls(controls)

        logging.info(f"[Камера {self.camera_id}] Обновлены параметры: {controls}")

    def apply_settings(self, master):
        """Копирует настройки ведущей камеры"""
        with master.exposure_lock:
            self.exposure_time = master.exposure_time
            self.analogue_gain = master.analogue_gain
            self.colour_gains = master.colour_gains
            self.contrast = master.contrast
            self.saturation = master.saturation
            self.sharpness = master.sharpness
            self.lens_position = master.lens_position

        self.update_needed.set()

    def get_frame(self):
        """Возвращает последний кадр"""
        return self.frame

    def stop_camera(self):
        """Останавливает потоки"""
        self.running = False
        self.update_needed.set()
        if self.update_thread:
            self.update_thread.join()

        if self.picam:
            self.picam.close()
        logging.info(f"Камера {self.camera_id} остановлена.")


class StereoCameraSystem:
    """Система стереокамер с полной синхронизацией параметров."""

    def __init__(self, camera0_id=0, camera1_id=1):
        self.cam0 = CameraDriver(camera_id=camera0_id, autofocus=True)
        self.cam1 = CameraDriver(camera_id=camera1_id, autofocus=False)

        self.cam0.auto_adjust = True
        self.cam1.auto_adjust = False

        logging.info("Система стереокамер инициализирована.")

    def start(self):
        """Запускает обе камеры"""
        self.cam0.start_camera()
        self.cam1.start_camera()
        logging.info("Обе камеры запущены.")

    def get_synchronized_frames(self):
        """Возвращает последние кадры с обеих камер"""
        self.cam1.apply_settings(self.cam0)
        return self.cam0.get_frame(), self.cam1.get_frame()

    def stop(self):
        """Останавливает обе камеры"""
        self.cam0.stop_camera()
        self.cam1.stop_camera()
        logging.info("Система стереокамер остановлена.")
