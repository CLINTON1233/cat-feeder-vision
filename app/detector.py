import cv2
from ultralytics import YOLO
from app.mqtt_client import send_feed

class CatDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

        self.target_labels = {
            "cat": (0, 255, 0),       # hijau
            "person": (0, 0, 255),    # merah
        }

        self.cat_detected = False
        self.person_detected = False

    def detect(self, frame):
        results = self.model(
            frame,
            conf=0.5,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        cat_found = False
        person_found = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                if label in self.target_labels:
                    color = self.target_labels[label]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

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

                    if label == "cat":
                        cat_found = True

                    if label == "person":
                        person_found = True

        # ===== CAT LOGIC =====
        if cat_found and not self.cat_detected:
            print("🐱 CAT DETECTED → SEND MQTT")
            send_feed("CAT")
            self.cat_detected = True

        if not cat_found:
            self.cat_detected = False

        # ===== PERSON LOGIC =====
        if person_found and not self.person_detected:
            print("🧍 PERSON DETECTED → SEND MQTT")
            send_feed("PERSON")
            self.person_detected = True

        if not person_found:
            self.person_detected = False

        return frame
