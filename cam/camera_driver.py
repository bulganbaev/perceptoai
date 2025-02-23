import libcamera
import cv2
import numpy as np
import threading
import atexit
from picamera2 import Picamera2


class CameraDriver:
    def __init__(self, camera_id=0, width=1920, height=1080, autofocus=True):
        self.camera_id = camera_id  # 0 или 1
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.thread = None
        self.autofocus = autofocus
        self.lens_position = None  # Локальная переменная для хранения фокуса

        try:
            self.picam = Picamera2(camera_id)
            controls = self.picam.camera_controls
            control_params = {}

            if "AfMode" in controls:
                control_params["AfMode"] = 2 if autofocus else 0  # Включаем автофокус
                control_params["AfSpeed"] = 1  # 1 – Fast (быстрая автофокусировка)

            # Создаем конфигурацию камеры
            config = self.picam.create_still_configuration(
                main={'size': (self.width, self.height)},
                controls=control_params
            )
            self.picam.configure(config)
        except Exception as e:
            print(f"Ошибка при инициализации камеры {camera_id}: {e}")
            self.picam = None  # Если камера недоступна, помечаем ее как None

    def start_camera(self):
        """Запускает поток захвата изображения"""
        if self.running or self.picam is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Основной цикл захвата изображений"""
        try:
            self.picam.start()
            while self.running:
                frame = self.picam.capture_array()
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Исправление BGR → RGB
        except Exception as e:
            print(f"Ошибка в потоке камеры {self.camera_id}: {e}")
        finally:
            self.picam.stop()

    def get_frame(self):
        """Возвращает последний кадр"""
        return self.frame

    def get_focus(self):
        """Получает текущее значение фокуса"""
        try:
            metadata = self.picam.capture_metadata()
            if "LensPosition" in metadata:
                self.lens_position = metadata["LensPosition"]
                return self.lens_position
        except Exception as e:
            print(f"Ошибка при получении LensPosition камеры {self.camera_id}: {e}")
        return None

    def set_focus(self, focus_value):
        """Применяет заданное значение фокуса"""
        try:
            if self.picam:
                self.picam.set_controls({"LensPosition": focus_value})
        except Exception as e:
            print(f"Ошибка при установке фокуса камеры {self.camera_id}: {e}")

    def stop_camera(self):
        """Останавливает поток захвата"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.picam:
            self.picam.close()


class StereoCameraSystem:
    def __init__(self, camera0_id=0, camera1_id=1):
        self.cam0 = CameraDriver(camera_id=camera0_id, autofocus=True)
        self.cam1 = CameraDriver(camera_id=camera1_id, autofocus=True)
        self.focus_lock = threading.Lock()  # Лок для синхронизации фокуса
        self.shared_lens_position = None  # Общий фокус

    def start(self):
        """Запускает обе камеры"""
        self.cam0.start_camera()
        self.cam1.start_camera()
        threading.Thread(target=self.sync_focus_loop, daemon=True).start()  # Запускаем поток синхронизации

    def sync_focus_loop(self):
        """Поток синхронизации фокуса"""
        while self.cam0.running and self.cam1.running:
            with self.focus_lock:
                focus0 = self.cam0.get_focus()
                if focus0 is not None:
                    self.shared_lens_position = focus0
                    self.cam1.set_focus(self.shared_lens_position)

    def get_frames(self):
        """Возвращает кадры с обеих камер"""
        return self.cam0.get_frame(), self.cam1.get_frame()

    def stop(self):
        """Останавливает обе камеры"""
        self.cam0.stop_camera()
        self.cam1.stop_camera()


if __name__ == "__main__":
    stereo_system = StereoCameraSystem()

    atexit.register(stereo_system.stop)

    stereo_system.start()

    try:
        while True:
            frame0, frame1 = stereo_system.get_frames()

            if frame0 is not None and frame1 is not None:
                combined = np.hstack((frame0, frame1))  # Объединяем два изображения
                cv2.namedWindow("Stereo Cameras", cv2.WINDOW_NORMAL)  # Делаем окно изменяемым
                cv2.resizeWindow("Stereo Cameras", 960, 540)
                cv2.imshow("Stereo Cameras", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nВыход по Ctrl+C")
    finally:
        stereo_system.stop()
        cv2.destroyAllWindows()
