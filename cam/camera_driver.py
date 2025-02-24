import libcamera
import cv2
import numpy as np
import threading
import atexit
import time
from picamera2 import Picamera2


class CameraDriver:
    """
    Драйвер для камеры OV5647 с возможностью синхронизации параметров экспозиции.
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

                # Только первая камера выполняет автонастройку экспозиции
                if self.auto_adjust and time.time() - self.last_adjust_time > 1:
                    self.adjust_exposure()
                    self.last_adjust_time = time.time()

        except Exception as e:
            print(f"Ошибка в потоке камеры {self.camera_id}: {e}")
        finally:
            self.picam.stop()

    def adjust_exposure(self):
        """Автоматически корректирует экспозицию на основе яркости кадра"""
        if self.frame is None:
            return

        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)

        metadata = self.picam.capture_metadata()
        self.exposure_time = metadata.get("ExposureTime", 1000)
        self.analogue_gain = metadata.get("AnalogueGain", 1.0)

        if avg_brightness < 50:  # Темно → увеличиваем экспозицию
            self.exposure_time = min(self.exposure_time * 1.5, 30000)
            self.analogue_gain = min(self.analogue_gain * 1.2, 4)
        elif avg_brightness > 180:  # Ярко → уменьшаем экспозицию
            self.exposure_time = max(self.exposure_time * 0.7, 1000)
            self.analogue_gain = max(self.analogue_gain * 0.8, 1)

        self.picam.set_controls({"AeEnable": 0, "ExposureTime": int(self.exposure_time), "AnalogueGain": self.analogue_gain})
        print(f"[Камера {self.camera_id}] Экспозиция: {int(self.exposure_time)} | Усиление: {self.analogue_gain:.2f}")

    def apply_exposure(self, exposure_time, analogue_gain):
        """Применяет переданные параметры экспозиции (для ведомой камеры)"""
        self.exposure_time = exposure_time
        self.analogue_gain = analogue_gain
        self.picam.set_controls({"AeEnable": 0, "ExposureTime": int(self.exposure_time), "AnalogueGain": self.analogue_gain})
        print(f"[Камера {self.camera_id}] Применены параметры ведущей камеры: {int(self.exposure_time)} | {self.analogue_gain:.2f}")

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
    """
    Система стереокамер с синхронизацией кадров и общей коррекцией экспозиции.
    """

    def __init__(self, camera0_id=0, camera1_id=1):
        self.cam0 = CameraDriver(camera_id=camera0_id, autofocus=True)
        self.cam1 = CameraDriver(camera_id=camera1_id, autofocus=True)

        # Включаем автонастройку экспозиции только для первой камеры (ведущей)
        self.cam0.auto_adjust = True
        self.cam1.auto_adjust = False  # Вторая камера копирует параметры первой

    def start(self):
        """Запускает обе камеры"""
        self.cam0.start_camera()
        self.cam1.start_camera()

    def get_synchronized_frames(self):
        """Ожидает и возвращает синхронизированные кадры с обеих камер"""
        frame0, frame1 = None, None
        with self.cam0.frame_ready, self.cam1.frame_ready:
            self.cam0.frame_ready.wait()
            self.cam1.frame_ready.wait()
            frame0 = self.cam0.get_frame()
            frame1 = self.cam1.get_frame()

        # Синхронизация экспозиции: ведомая камера копирует параметры ведущей
        self.cam1.apply_exposure(self.cam0.exposure_time, self.cam0.analogue_gain)

        return frame0, frame1

    def stop(self):
        """Останавливает обе камеры"""
        self.cam0.stop_camera()
        self.cam1.stop_camera()


if __name__ == "__main__":
    # Инициализация системы стереокамер
    stereo_system = StereoCameraSystem()

    # Регистрация выхода
    atexit.register(stereo_system.stop)

    # Запуск системы
    stereo_system.start()

    print("Нажмите 'q' для выхода.")

    try:
        while True:
            frame0, frame1 = stereo_system.get_synchronized_frames()

            if frame0 is not None and frame1 is not None:
                combined = np.hstack((frame0, frame1))  # Объединение двух изображений
                cv2.namedWindow("Stereo Camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Stereo Camera", 960, 540)
                cv2.imshow("Stereo Camera", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nВыход по Ctrl+C")
    finally:
        stereo_system.stop()
        cv2.destroyAllWindows()
