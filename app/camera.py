import cv2
import os

class Camera:
    def __init__(self):
        self.cap = None

        video_devices = [f"/dev/{d}" for d in os.listdir("/dev") if d.startswith("video")]

        for dev in video_devices:
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                print(f"📷 Camera found at {dev}")
                self.cap = cap
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return
            cap.release()

        raise RuntimeError("No camera detected")

    def get_frame(self):
        success, frame = self.cap.read()
        return frame if success else None

    def release(self):
        if self.cap:
            self.cap.release()
