import libcamera
import cv2
import numpy as np
import threading
import atexit
import time  # Для задержек и временных параметров
from picamera2 import Picamera2


class CameraDriver:
    """
    Драйвер для работы с одной камерой.
    Инициализирует Picamera2, настраивает конфигурацию, запускает поток захвата изображений,
    обеспечивает управление автофокусом и параметрами экспозиции.
    Дополнительно позволяет переключаться между режимами освещенности (indoor / outdoor).
    """

    def __init__(self, camera_id=0, width=1920, height=1080, autofocus=True,
                 manual_exposure=False, exposure_time=None, analogue_gain=None):
        self.camera_id = camera_id  # Идентификатор камеры (например, 0 или 1)
        self.width = width
        self.height = height
        self.running = False  # Флаг работы камеры
        self.frame = None  # Последний захваченный кадр
        self.thread = None  # Поток для захвата изображений
        self.autofocus = autofocus
        self.lens_position = None  # Текущее значение фокуса
        self.manual_exposure = manual_exposure  # Режим ручной экспозиции
        self.exposure_time = exposure_time  # Время экспозиции (при ручном режиме)
        self.analogue_gain = analogue_gain  # Значение аналогового усиления (при ручном режиме)

        try:
            # Инициализация камеры с использованием Picamera2
            self.picam = Picamera2(camera_id)
            controls = self.picam.camera_controls
            control_params = {}

            # Настройка автофокуса, если поддерживается камерой
            if "AfMode" in controls:
                control_params["AfMode"] = 2 if autofocus else 0  # 2 – автофокус включен, 0 – выключен
                control_params["AfSpeed"] = 1  # 1 – быстрая автофокусировка

            # Настройка экспозиции: если включен ручной режим, отключаем автоматическую экспозицию
            if self.manual_exposure:
                if "AeEnable" in controls:
                    control_params["AeEnable"] = 0
                if exposure_time is not None and "ExposureTime" in controls:
                    control_params["ExposureTime"] = exposure_time
                if analogue_gain is not None and "AnalogueGain" in controls:
                    control_params["AnalogueGain"] = analogue_gain
            else:
                # Автоматическая экспозиция, если ручной режим не выбран
                if "AeEnable" in controls:
                    control_params["AeEnable"] = 1

            # Создаем конфигурацию для захвата изображения с указанным разрешением и параметрами
            config = self.picam.create_still_configuration(
                main={'size': (self.width, self.height)},
                controls=control_params
            )
            self.picam.configure(config)
        except Exception as e:
            print(f"Ошибка при инициализации камеры {camera_id}: {e}")
            self.picam = None  # Если камера не доступна, помечаем её как None

    def start_camera(self):
        """
        Запускает поток захвата изображений, если камера успешно инициализирована.
        """
        if self.running or self.picam is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """
        Основной цикл захвата изображений.
        Захватывает кадры, конвертирует из BGR в RGB и сохраняет последний кадр.
        """
        try:
            self.picam.start()
            while self.running:
                frame = self.picam.capture_array()  # Захват кадра в формате NumPy
                # Конвертация изображения из BGR (формат OpenCV) в RGB
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Ошибка в потоке камеры {self.camera_id}: {e}")
        finally:
            self.picam.stop()

    def get_frame(self):
        """
        Возвращает последний захваченный кадр.
        """
        return self.frame

    def get_focus(self):
        """
        Получает текущее значение фокуса (LensPosition) из метаданных камеры.
        """
        try:
            metadata = self.picam.capture_metadata()
            if "LensPosition" in metadata:
                self.lens_position = metadata["LensPosition"]
                return self.lens_position
        except Exception as e:
            print(f"Ошибка при получении LensPosition камеры {self.camera_id}: {e}")
        return None

    def set_focus(self, focus_value):
        """
        Устанавливает заданное значение фокуса для камеры.
        :param focus_value: значение для установки позиции объектива
        """
        try:
            if self.picam:
                self.picam.set_controls({"LensPosition": focus_value})
        except Exception as e:
            print(f"Ошибка при установке фокуса камеры {self.camera_id}: {e}")

    def set_exposure(self, exposure_time=None, analogue_gain=None):
        """
        Устанавливает ручные параметры экспозиции для камеры.
        Отключает автоматическую экспозицию и применяет заданные значения.
        :param exposure_time: время экспозиции (например, в микросекундах)
        :param analogue_gain: значение аналогового усиления
        """
        try:
            if self.picam:
                controls = {"AeEnable": 0}  # Отключаем автоматическую экспозицию
                if exposure_time is not None:
                    controls["ExposureTime"] = exposure_time
                    self.exposure_time = exposure_time
                if analogue_gain is not None:
                    controls["AnalogueGain"] = analogue_gain
                    self.analogue_gain = analogue_gain
                self.picam.set_controls(controls)
                print(
                    f"Установлены ручные параметры экспозиции: ExposureTime={exposure_time}, AnalogueGain={analogue_gain}")
        except Exception as e:
            print(f"Ошибка при установке экспозиции камеры {self.camera_id}: {e}")

    def get_exposure(self):
        """
        Получает текущие параметры экспозиции из метаданных камеры (например, ExposureTime и AnalogueGain).
        :return: словарь с параметрами экспозиции или None, если данные не получены
        """
        try:
            metadata = self.picam.capture_metadata()
            exposure_info = {}
            if "ExposureTime" in metadata:
                exposure_info["ExposureTime"] = metadata["ExposureTime"]
            if "AnalogueGain" in metadata:
                exposure_info["AnalogueGain"] = metadata["AnalogueGain"]
            return exposure_info if exposure_info else None
        except Exception as e:
            print(f"Ошибка при получении экспозиции камеры {self.camera_id}: {e}")
        return None

    def set_light_mode(self, mode):
        """
        Переключает режим работы камеры в зависимости от условий освещенности.
        Режим "indoor" (в помещении) – увеличенная экспозиция и усиление для работы при низкой освещенности.
        Режим "outdoor" (на улице) – уменьшенная экспозиция и низкое усиление для ярких условий.

        :param mode: строка "indoor" или "outdoor"
        """
        if mode == "indoor":
            # Пример: устанавливаем более длительную экспозицию и увеличенное усиление
            self.set_exposure(exposure_time=10000, analogue_gain=2)
            print("Установлен режим 'indoor' - для работы в помещении.")
        elif mode == "outdoor":
            # Пример: устанавливаем короткую экспозицию и минимальное усиление
            self.set_exposure(exposure_time=1000, analogue_gain=1)
            print("Установлен режим 'outdoor' - для работы на улице.")
        else:
            print(f"Неизвестный режим освещенности: {mode}")

    def stop_camera(self):
        """
        Останавливает поток захвата изображений и закрывает камеру.
        """
        self.running = False
        if self.thread:
            self.thread.join()
        if self.picam:
            self.picam.close()


class StereoCameraSystem:
    """
    Система стереокамер для синхронизации двух камер.
    Обеспечивает одновременный захват изображений, синхронизацию фокуса и переключение режимов освещенности.
    """

    def __init__(self, camera0_id=0, camera1_id=1):
        # Инициализация двух камер с включенным автофокусом
        self.cam0 = CameraDriver(camera_id=camera0_id, autofocus=True)
        self.cam1 = CameraDriver(camera_id=camera1_id, autofocus=True)
        self.focus_lock = threading.Lock()  # Лок для синхронизации фокуса
        self.shared_lens_position = None  # Общее значение фокуса для синхронизации

    def start(self):
        """
        Запускает обе камеры и поток синхронизации фокуса.
        """
        self.cam0.start_camera()
        self.cam1.start_camera()
        threading.Thread(target=self.sync_focus_loop, daemon=True).start()

    def sync_focus_loop(self):
        """
        Поток синхронизации фокуса между камерами.
        Каждые 50 миллисекунд получает фокус с первой камеры и применяет его ко второй.
        """
        while self.cam0.running and self.cam1.running:
            with self.focus_lock:
                focus0 = self.cam0.get_focus()
                if focus0 is not None:
                    self.shared_lens_position = focus0
                    self.cam1.set_focus(self.shared_lens_position)
            time.sleep(0.05)

    def set_light_mode(self, mode):
        """
        Устанавливает выбранный режим освещенности ("indoor" или "outdoor") для обеих камер.
        :param mode: строка "indoor" или "outdoor"
        """
        self.cam0.set_light_mode(mode)
        self.cam1.set_light_mode(mode)

    def get_frames(self):
        """
        Возвращает последние захваченные кадры с обеих камер.
        :return: кортеж (frame0, frame1)
        """
        return self.cam0.get_frame(), self.cam1.get_frame()

    def stop(self):
        """
        Останавливает работу обеих камер.
        """
        self.cam0.stop_camera()
        self.cam1.stop_camera()


if __name__ == "__main__":
    # Инициализация системы стереокамер (OV5647)
    stereo_system = StereoCameraSystem()

    # Регистрация функции остановки камер при завершении работы программы
    atexit.register(stereo_system.stop)

    # Запуск системы камер
    stereo_system.start()

    print("Нажмите 'i' для режима indoor (в помещении) или 'o' для режима outdoor (на улице).")
    print("Нажмите 'q' для выхода из программы.")

    try:
        while True:
            frame0, frame1 = stereo_system.get_frames()
            if frame0 is not None and frame1 is not None:
                combined = np.hstack((frame0, frame1))  # Горизонтальное объединение изображений
                cv2.namedWindow("Stereo Cameras", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Stereo Cameras", 960, 540)
                cv2.imshow("Stereo Cameras", combined)

            # Обработка нажатия клавиш:
            # 'q' — выход из программы
            # 'i' — установка режима indoor (для работы в помещении)
            # 'o' — установка режима outdoor (для работы на улице)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                stereo_system.set_light_mode("indoor")
            elif key == ord('o'):
                stereo_system.set_light_mode("outdoor")
    except KeyboardInterrupt:
        print("\nВыход по Ctrl+C")
    finally:
        stereo_system.stop()
        cv2.destroyAllWindows()
