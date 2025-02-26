from cam.camera_driver import StereoCameraSystem
import cv2
import degirum as dg

model = dg.load_model(
    model_name='yolov8m_seg',
    inference_host_address='@local',
    zoo_url='/home/drone/git/model_zoo/yolov8m_seg/yolov8m_seg.json'
)


stereo = StereoCameraSystem()
stereo.start()



try:
    while True:
        frame_left, frame_right = stereo.get_synchronized_frames()
        if frame_left is None:
            continue
        inference_result = model(frame_left)
        cv2.imshow("Segmentation Output", inference_result.image_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("⏹️ Остановка потока...")


stereo.stop()
cv2.destroyAllWindows()
print("✅ Поток завершён.")
