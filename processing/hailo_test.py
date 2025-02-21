import cv2
import hailo_detection as hd

# Инициализация модели YOLOv11s
inf = hd.HailoInference('data/models/yolov11s.hef', 'data/labels/coco.txt')
proc = hd.Processor(inf, 0.5)

# Загружаем изображение
img = cv2.imread('data/images/left/left_00.jpg')

# Обрабатываем изображение через модель
detected = proc.process([img])


