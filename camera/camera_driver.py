import libcamera
import time
import threading
import cv2
import numpy as np


class CameraDriver:
    def __init__(self, camera_id=0, width=1920, height=1080):
        self.camera_id = camera_id  # 0 или 1 (порт камеры)
        self.width = width
        self.height = height
        self.running = False
        self.frame = None
        self.thread = None

    def start_camera(self):
        """Запускает поток захвата изображения"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.start()

    def _capture_loop(self):
        """Основной цикл захвата изображений"""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print(f"Ошибка: Камера {self.camera_id} не запустилась!")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame = frame
            time.sleep(0.01)  # Уменьшает нагрузку на CPU
        cap.release()

    def get_frame(self):
        """Возвращает последний кадр"""
        return self.frame

    def stop_camera(self):
        """Останавливает поток захвата"""
        self.running = False
        if self.thread:
            self.thread.join()


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
