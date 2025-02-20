import cv2
import hailo_detection as hd


inf = hd.HailoInference('data/models/yolov11s.hef')
proc = hd.Processor(inf, 0.5)

img = cv2.imread('data/images/test_image.jpg')
detected = proc.process([img])
# print(f'DETECTED {detected}')
print(f'DETECTED {detected}')
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # Делаем окно изменяемым
cv2.resizeWindow("Camera", 960, 540)
cv2.imshow("Camera", img)



