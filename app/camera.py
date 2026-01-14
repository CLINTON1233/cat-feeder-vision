import cv2
import os
import time

class Camera:
    def __init__(self):
        self.cap = None

        video_devices = sorted([f"/dev/{d}" for d in os.listdir("/dev") if d.startswith("video")])
        print("🔍 Found devices:", video_devices)

        for dev in video_devices:
            print(f"🔄 Trying camera: {dev}")
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)  # paksa backend V4L2

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            time.sleep(1)  # kasih waktu kamera bangun

            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"📷 Camera ACTIVE at {dev}")
                    self.cap = cap
                    return
                else:
                    print(f"⚠️ Opened but no frame from {dev}")

            cap.release()

        raise RuntimeError("❌ No working camera detected")

    def get_frame(self):
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
