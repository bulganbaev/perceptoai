import libcamera
import cv2
import numpy as np
import threading
import atexit
import time
from picamera2 import Picamera2


class CameraDriver:
    """
    Драйвер для камеры OV5647 с полной синхронизацией параметров.
    """

    def __init__(self, camera_id=0, width=1920, height=1080, autofocus=True):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.autofocus = autofocus
        self.auto_adjust = False  # Только первая камера выполняет коррекцию
        self.last_adjust_time = time.time()
        self.exposure_time = 1000
        self.analogue_gain = 1.0
        self.colour_gains = (1.0, 1.0)  # Баланс белого (R, B)
        self.contrast = 1.0
        self.saturation = 1.0
        self.sharpness = 1.0
        self.frame_ready = threading.Condition()  # Синхронизация кадров

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

        except Exception as e:
            print(f"Ошибка при инициализации камеры {camera_id}: {e}")
            self.picam = None

    def start_camera(self):
        """Запускает поток захвата изображений"""
        if self.running or self.picam is None:
            return
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Основной цикл захвата изображений"""
        try:
            self.picam.start()
            while self.running:
                frame = self.picam.capture_array()
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Синхронизация кадров
                with self.frame_ready:
                    self.frame_ready.notify_all()

                # Только первая камера выполняет автонастройку параметров
                if self.auto_adjust and time.time() - self.last_adjust_time > 1:
                    self.adjust_exposure()
                    self.last_adjust_time = time.time()

        except Exception as e:
            print(f"Ошибка в потоке камеры {self.camera_id}: {e}")
        finally:
            self.picam.stop()

    def adjust_exposure(self):
        """Автоматически корректирует экспозицию и другие параметры"""
        if self.frame is None:
            return

        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)

        metadata = self.picam.capture_metadata()
        self.exposure_time = metadata.get("ExposureTime", 1000)
        self.analogue_gain = metadata.get("AnalogueGain", 1.0)
        self.colour_gains = metadata.get("ColourGains", (1.0, 1.0))
        self.contrast = metadata.get("Contrast", 1.0)
        self.saturation = metadata.get("Saturation", 1.0)
        self.sharpness = metadata.get("Sharpness", 1.0)

        if avg_brightness < 50:  # Темно → увеличиваем экспозицию
            self.exposure_time = min(self.exposure_time * 1.5, 30000)
            self.analogue_gain = min(self.analogue_gain * 1.2, 4)
        elif avg_brightness > 180:  # Ярко → уменьшаем экспозицию
            self.exposure_time = max(self.exposure_time * 0.7, 1000)
            self.analogue_gain = max(self.analogue_gain * 0.8, 1)

        self.picam.set_controls({
            "AeEnable": 0,
            "ExposureTime": int(self.exposure_time),
            "AnalogueGain": self.analogue_gain,
            "ColourGains": self.colour_gains,
            "Contrast": self.contrast,
            "Saturation": self.saturation,
            "Sharpness": self.sharpness,
        })

    def apply_settings(self, master):
        """Применяет настройки ведущей камеры"""
        self.picam.set_controls({
            "AeEnable": 0,
            "ExposureTime": int(master.exposure_time),
            "AnalogueGain": master.analogue_gain,
            "ColourGains": master.colour_gains,
            "Contrast": master.contrast,
            "Saturation": master.saturation,
            "Sharpness": master.sharpness,
        })

    def get_frame(self):
        """Ждет, пока будет доступен новый кадр, и возвращает его"""
        with self.frame_ready:
            self.frame_ready.wait()
            return self.frame

    def stop_camera(self):
        """Останавливает поток захвата"""
        self.running = False
        if self.picam:
            self.picam.close()


class StereoCameraSystem:
    """Система стереокамер с полной синхронизацией параметров."""

    def __init__(self, camera0_id=0, camera1_id=1):
        self.cam0 = CameraDriver(camera_id=camera0_id, autofocus=True)
        self.cam1 = CameraDriver(camera_id=camera1_id, autofocus=True)

        self.cam0.auto_adjust = True  # Только первая камера регулирует параметры
        self.cam1.auto_adjust = False

    def start(self):
        """Запускает обе камеры"""
        self.cam0.start_camera()
        self.cam1.start_camera()

    def get_synchronized_frames(self):
        """Ожидает и возвращает синхронизированные кадры"""
        frame0, frame1 = self.cam0.get_frame(), self.cam1.get_frame()
        self.cam1.apply_settings(self.cam0)
        return frame0, frame1

    def stop(self):
        """Останавливает обе камеры"""
        self.cam0.stop_camera()
        self.cam1.stop_camera()
