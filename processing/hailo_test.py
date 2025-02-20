import cv2
import hailo_detection as hd

# Инициализация модели YOLOv11s
inf = hd.HailoInference('data/models/yolov11s.hef')
proc = hd.Processor(inf, 0.5)

# Загружаем изображение
img = cv2.imread('data/images/test_image.jpg')

# Обрабатываем изображение через модель
detected = proc.process([img])

# Отрисовываем боксы
for result in detected:
    for box in result["absolute_boxes"]:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелёный цвет, толщина 2
        cv2.putText(img, "Object", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Показываем изображение с бокcами
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 960, 540)
cv2.imshow("Camera", img)

# Ожидаем нажатие клавиши и закрываем окно
cv2.waitKey(0)
cv2.destroyAllWindows()
