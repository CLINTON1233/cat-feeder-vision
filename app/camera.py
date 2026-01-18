import cv2
import os
import time
import threading
import queue
import numpy as np

class Camera:
    def __init__(self):
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)  # Buffer kecil
        self.running = False
        self.capture_thread = None
        
        video_devices = sorted([f"/dev/{d}" for d in os.listdir("/dev") if d.startswith("video")])
        print("🔍 Found devices:", video_devices)

        for dev in video_devices:
            print(f"🔄 Trying camera: {dev}")
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            
            # OPTIMAL SETTINGS UNTUK SMOOTHNESS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)           # Frame rate
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # Buffer kecil untuk mengurangi latency
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Codec MJPG
            
            # Optimasi untuk performa
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # Matikan autofocus
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure 1 = manual mode
            cap.set(cv2.CAP_PROP_EXPOSURE, 100)     # Exposure value

            time.sleep(0.5)  # Waktu stabilisasi lebih singkat

            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"📷 Camera ACTIVE at {dev}")
                    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                    self.cap = cap
                    
                    # Mulai thread capture
                    self.start_capture_thread()
                    return
                else:
                    print(f"⚠️ Opened but no frame from {dev}")

            cap.release()

        raise RuntimeError("❌ No working camera detected")
    
    def start_capture_thread(self):
        """Mulai thread terpisah untuk capture frame"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print("🎬 Capture thread started")
    
    def _capture_frames(self):
        """Thread untuk menangkap frame secara kontinu"""
        frame_count = 0
        fps_timer = time.time()
        
        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame")
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                
                # Hitung FPS setiap 100 frame
                if frame_count % 100 == 0:
                    elapsed = time.time() - fps_timer
                    fps = 100 / elapsed
                    fps_timer = time.time()
                    # print(f"📊 Capture FPS: {fps:.1f}")
                
                # Hanya simpan frame terbaru (skip jika queue penuh)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Buang frame lama
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame.copy(), block=False)
                
                # Small delay untuk mengurangi CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                print(f"⚠️ Capture error: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Ambil frame terbaru dari queue"""
        if not self.running or self.frame_queue.empty():
            return None
        
        try:
            # Ambil frame terbaru tanpa blocking
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop camera dan cleanup"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        print("📷 Camera stopped")
    
    def release(self):
        """Alias untuk stop"""
        self.stop()