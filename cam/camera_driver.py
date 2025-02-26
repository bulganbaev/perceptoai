import cv2
import numpy as np
import threading
import time
import logging
from libcamera import controls
from picamera2 import Picamera2

# Настройки логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("stereo_camera.log"), logging.StreamHandler()]
)


class CameraDriver:
    """
    Драйвер для камеры OV5647 с синхронизацией экспозиции и автофокусировки.
    """

    def __init__(self, camera_id=0, width=1920, height=1080, autofocus=True):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.autofocus = autofocus
        self.auto_adjust = False
        self.exposure_lock = threading.Lock()
        self.update_needed = threading.Event()

        # Начальные настройки
        self.exposure_time = 10000
        self.analogue_gain = 1.0
        self.colour_gains = (1.0, 1.0)
        self.contrast = 1.0
        self.saturation = 1.0
        self.sharpness = 1.0
        self.lens_position = None

        try:
            self.picam = Picamera2(camera_id)
            controls_list = self.picam.camera_controls
            control_params = {}

            # Автофокус, если поддерживается
            if "AfMode" in controls_list:
                control_params["AfMode"] = controls.AfModeEnum.Continuous if autofocus else controls.AfModeEnum.Manual
                control_params["AfSpeed"] = controls.AfSpeedEnum.Fast  # Быстрая автофокусировка
                control_params["AfRange"] = controls.AfRangeEnum.Full  # Полный диапазон

            # Создание конфигурации камеры
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
        threading.Thread(target=self._apply_settings_loop, daemon=True).start()
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

    def adjust_exposure(self):
        """Автоматически корректирует экспозицию и яркость"""
        if self.frame is None:
            return

        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)

        metadata = self.picam.capture_metadata()
        with self.exposure_lock:
            self.exposure_time = metadata.get("ExposureTime", 10000)
            self.analogue_gain = metadata.get("AnalogueGain", 1.0)

            if avg_brightness < 50:
                self.exposure_time = min(self.exposure_time * 1.5, 600000)
                self.analogue_gain = min(self.analogue_gain * 1.2, 4)
            elif avg_brightness > 180:
                self.exposure_time = max(self.exposure_time * 0.7, 1000)
                self.analogue_gain = max(self.analogue_gain * 0.8, 1)

            self.update_needed.set()

        logging.warning(f'{avg_brightness=}')
        logging.info(f"[Камера {self.camera_id}] Обновление экспозиции: {self.exposure_time}, {self.analogue_gain}")

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
        self.picam.stop()
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


# --- Тестирование ---
if __name__ == "__main__":
    stereo = StereoCameraSystem()
    stereo.start()
    time.sleep(3)

    frame_left, frame_right = stereo.get_synchronized_frames()
    if frame_left is not None and frame_right is not None:
        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    stereo.stop()
