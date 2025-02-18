import libcamera
import cv2
import numpy as np
import threading
from picamera2 import Picamera2

class CameraDriver:
    def __init__(self, camera_id=0, width=1920, height=1080, autofocus=True):
        self.camera_id = camera_id  # 0 или 1
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.thread = None
        self.picam = Picamera2(camera_id)

        config = self.picam.create_still_configuration(main={'size': (self.width, self.height)},
                                                       controls={"AfMode": 2 if autofocus else 0})
        self.picam.configure(config)

    def start_camera(self):
        """Запускает поток захвата изображения"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()

    def _capture_loop(self):
        """Основной цикл захвата изображений"""
        self.picam.start()
        while self.running:
            frame = self.picam.capture_array()
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Исправление BGR → RGB
        self.picam.stop()

    def get_frame(self):
        """Возвращает последний кадр"""
        return self.frame

    def stop_camera(self):
        """Останавливает поток захвата"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.picam.close()

if __name__ == "__main__":
    cam0 = CameraDriver(camera_id=0)
    cam1 = CameraDriver(camera_id=1)

    cam0.start_camera()
    cam1.start_camera()

    try:
        while True:
            frame0 = cam0.get_frame()
            frame1 = cam1.get_frame()

            if frame0 is not None and frame1 is not None:
                combined = np.hstack((frame0, frame1))  # Объединяем два изображения
                cv2.imshow("Dual Cameras", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    cam0.stop_camera()
    cam1.stop_camera()
    cv2.destroyAllWindows()