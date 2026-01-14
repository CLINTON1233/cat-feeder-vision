import cv2
from ultralytics import YOLO
from app.mqtt_client import send_feed
import time

class CatDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

        self.target_labels = {
            "cat": (0, 255, 0),
            "person": (0, 0, 255),
        }

        self.last_send = 0
        self.frame_count = 0

        self.detect_interval = 100   # tetap 100
        self.box_timeout = 5         # DIPERKECIL → biar gak lengket

        self.last_boxes = []         # [(label, color, conf, x1,y1,x2,y2, age)]

    def detect(self, frame):
        self.frame_count += 1

        # ===== AGING BOX =====
        new_boxes = []
        for b in self.last_boxes:
            label, color, conf, x1, y1, x2, y2, age = b
            age += 1
            if age < self.box_timeout:
                new_boxes.append((label, color, conf, x1, y1, x2, y2, age))
        self.last_boxes = new_boxes

        # ===== RUN YOLO =====
        if self.frame_count % self.detect_interval == 0:
            results = self.model(
                frame,
                conf=0.5,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            cat_found = False
            person_found = False

            # 🔥 RESET BOX LAMA → BIKIN PRESISI & TIDAK LENGKET
            self.last_boxes = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]

                    if label in self.target_labels:
                        color = self.target_labels[label]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        # langsung replace box lama
                        self.last_boxes.append(
                            (label, color, conf, x1, y1, x2, y2, 0)
                        )

                        if label == "cat":
                            cat_found = True
                        if label == "person":
                            person_found = True

            # ===== MQTT ANTI SPAM =====
            now = time.time()
            if now - self.last_send > 5:
                if cat_found:
                    print("🐱 CAT DETECTED → SEND MQTT")
                    send_feed("CAT")
                    self.last_send = now
                elif person_found:
                    print("🧍 PERSON DETECTED → SEND MQTT")
                    send_feed("PERSON")
                    self.last_send = now

        # ===== DRAW =====
        for label, color, conf, x1, y1, x2, y2, age in self.last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label.upper()} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame
