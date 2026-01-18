import cv2
from ultralytics import YOLO
from app.mqtt_client import send_feed
import time
import numpy as np

class CatDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        
        self.target_labels = {
            "cat": (0, 255, 0),     # Hijau untuk kucing
            "person": (0, 0, 255),  # Merah untuk orang
        }
        
        self.last_send = 0
        self.frame_count = 0
        
        # Interval deteksi
        self.detect_interval = 5
        self.box_timeout = 15
        
        # Tracking variables
        self.tracked_boxes = {}
        self.next_track_id = 0
        
        # NEW: State tracking
        self.last_detection_state = {"cat": False, "person": False}
        self.current_detection_state = {"cat": False, "person": False}
        self.new_detection_flag = {"cat": False, "person": False}
        
        self.class_names = self.model.names
        
    def _calculate_iou(self, box1, box2):
        """Menghitung Intersection over Union antara dua bounding box"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Area masing-masing box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Koordinat overlap
        x1_overlap = max(x1_1, x1_2)
        y1_overlap = max(y1_1, y1_2)
        x2_overlap = min(x2_1, x2_2)
        y2_overlap = min(y2_1, y2_2)
        
        # Area overlap
        overlap_width = max(0, x2_overlap - x1_overlap)
        overlap_height = max(0, y2_overlap - y1_overlap)
        overlap_area = overlap_width * overlap_height
        
        # IoU
        iou = overlap_area / (area1 + area2 - overlap_area + 1e-6)
        return iou
    
    def _assign_track_ids(self, current_detections):
        """Meng-assign track ID ke deteksi baru berdasarkan IoU"""
        updated_boxes = {}
        used_track_ids = set()
        
        # Untuk setiap deteksi baru
        for det in current_detections:
            label, color, conf, x1, y1, x2, y2 = det
            new_box = (x1, y1, x2, y2)
            matched = False
            
            # Cari track ID yang cocok berdasarkan IoU
            for track_id, (old_label, old_color, old_conf, ox1, oy1, ox2, oy2, age, _) in self.tracked_boxes.items():
                if old_label != label:
                    continue  # Hanya match dengan label yang sama
                
                old_box = (ox1, oy1, ox2, oy2)
                iou = self._calculate_iou(new_box, old_box)
                
                # Jika IoU cukup besar, update track yang sudah ada
                if iou > 0.3:  # Threshold IoU
                    updated_boxes[track_id] = (label, color, conf, x1, y1, x2, y2, 0, track_id)
                    used_track_ids.add(track_id)
                    matched = True
                    break
            
            # Jika tidak ada yang match, buat track ID baru
            if not matched:
                new_track_id = self.next_track_id
                updated_boxes[new_track_id] = (label, color, conf, x1, y1, x2, y2, 0, new_track_id)
                used_track_ids.add(new_track_id)
                self.next_track_id += 1
        
        # Tambahkan boxes lama yang tidak dideteksi lagi (bertambah age-nya)
        for track_id, (label, color, conf, x1, y1, x2, y2, age, tid) in self.tracked_boxes.items():
            if track_id not in used_track_ids:
                age += 1
                if age < self.box_timeout:
                    updated_boxes[track_id] = (label, color, conf, x1, y1, x2, y2, age, tid)
        
        return updated_boxes
    
    def detect(self, frame):
        self.frame_count += 1
        
        # ===== RUN YOLO =====
        current_detections = []  # List untuk deteksi baru
        
        if self.frame_count % self.detect_interval == 0:
            results = self.model(
                frame,
                conf=0.5,
                imgsz=320, 
                device="cpu",
                verbose=False,
                half=False
            )
            
            # Kumpulkan semua deteksi baru
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.class_names[cls_id]
                    
                    if label in self.target_labels:
                        color = self.target_labels[label]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        current_detections.append((label, color, conf, x1, y1, x2, y2))
            
            # Update tracked boxes dengan track ID
            self.tracked_boxes = self._assign_track_ids(current_detections)
            
            # ===== NEW: Update detection states =====
            self.current_detection_state = {"cat": False, "person": False}
            
            for box_data in self.tracked_boxes.values():
                label = box_data[0]
                if label == "cat":
                    self.current_detection_state["cat"] = True
                elif label == "person":
                    self.current_detection_state["person"] = True
            
            # ===== NEW: Check for NEW detections =====
            self.new_detection_flag = {"cat": False, "person": False}
            
            # Check for NEW cat detection
            if self.current_detection_state["cat"] and not self.last_detection_state["cat"]:
                self.new_detection_flag["cat"] = True
                print("🐱 NEW CAT DETECTED!")
            
            # Check for NEW person detection
            if self.current_detection_state["person"] and not self.last_detection_state["person"]:
                self.new_detection_flag["person"] = True
                print("🧍 NEW PERSON DETECTED!")
            
            # ===== NEW: MQTT Logic hanya untuk DETECTION BARU =====
            now = time.time()
            
            # Kirim MQTT hanya jika ada DETECTION BARU
            if self.new_detection_flag["cat"]:
                if now - self.last_send > 5:  # Minimal 5 detik antar pengiriman
                    print(f"🐱 NEW CAT DETECTED → SEND MQTT")
                    send_feed("CAT")
                    self.last_send = now
            
            elif self.new_detection_flag["person"]:
                if now - self.last_send > 5:  # Minimal 5 detik antar pengiriman
                    print(f"🧍 NEW PERSON DETECTED → SEND MQTT")
                    send_feed("PERSON")
                    self.last_send = now
            
            # Update last detection state untuk frame berikutnya
            self.last_detection_state = self.current_detection_state.copy()
        
        # ===== DRAW ALL BOXES (SETIAP FRAME) =====
        for track_id, (label, color, conf, x1, y1, x2, y2, age, tid) in self.tracked_boxes.items():
            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Tambahkan label dengan confidence dan track ID
            label_text = f"{label.upper()} {conf:.2f} ID:{tid}"
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # NEW: Tampilkan status deteksi
        cat_count = sum(1 for data in self.tracked_boxes.values() if data[0] == "cat")
        person_count = sum(1 for data in self.tracked_boxes.values() if data[0] == "person")
        
        status_text = f"Cats: {cat_count} | Persons: {person_count}"
        
        # Warna berbeda untuk deteksi baru
        if self.new_detection_flag["cat"] or self.new_detection_flag["person"]:
            cv2.putText(
                frame,
                "NEW DETECTION!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),  # Kuning untuk deteksi baru
                2
            )
            cv2.putText(
                frame,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        return frame
