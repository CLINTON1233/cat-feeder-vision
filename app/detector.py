import cv2
from ultralytics import YOLO
from app.mqtt_client import send_feed
import time
import numpy as np

class CatDetector:
    def __init__(self):
        # Gunakan model yang lebih cepat
        self.model = YOLO("yolov8n.pt")
        
        print("=" * 50)
        print("🐱 CAT DETECTOR INITIALIZED")
        print("=" * 50)
        
        # Warna untuk kucing
        self.cat_color = (0, 255, 0)  # Hijau
        
        self.last_send = 0
        self.frame_count = 0
        self.last_cat_detection = 0
        
        # Tracking untuk smoothness
        self.tracked_boxes = {}
        self.next_track_id = 0
        self.box_timeout = 15
        
        # Filter untuk smooth bounding box
        self.box_filters = {}  # {track_id: KalmanFilter}
        
        # Performance tracking
        self.processing_times = []
        self.last_fps_update = time.time()
        
        print("✅ Detector ready - Only cats will be detected")
        print("=" * 50)
        
    def _calculate_iou(self, box1, box2):
        """Menghitung Intersection over Union (IOU)"""
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
        
        # IOU
        iou = overlap_area / (area1 + area2 - overlap_area + 1e-6)
        return iou
    
    def _smooth_box(self, track_id, box):
        """Smooth bounding box dengan moving average"""
        if track_id not in self.tracked_boxes:
            return box
        
        # Ambil history box
        history = self.tracked_boxes[track_id].get('history', [])
        history.append(box)
        
        # Simpan maks 5 history
        if len(history) > 5:
            history.pop(0)
        
        # Hitung moving average
        if len(history) >= 3:
            avg_box = np.mean(history, axis=0).astype(int)
            return avg_box
        
        return box
    
    def _assign_track_ids(self, current_detections):
        """Assign track IDs dengan IOU matching"""
        updated_boxes = {}
        used_track_ids = set()
        
        for det in current_detections:
            label, conf, x1, y1, x2, y2 = det
            new_box = (x1, y1, x2, y2)
            matched = False
            
            for track_id, data in self.tracked_boxes.items():
                if data['label'] != label:
                    continue
                
                old_box = (data['x1'], data['y1'], data['x2'], data['y2'])
                iou = self._calculate_iou(new_box, old_box)
                
                if iou > 0.4:  # Threshold untuk matching
                    # Update dengan smooth transition
                    smooth_box = self._smooth_box(track_id, (x1, y1, x2, y2))
                    x1_s, y1_s, x2_s, y2_s = smooth_box
                    
                    updated_boxes[track_id] = {
                        'label': label,
                        'conf': conf,
                        'x1': x1_s, 'y1': y1_s,
                        'x2': x2_s, 'y2': y2_s,
                        'age': 0,
                        'history': data.get('history', []) + [(x1, y1, x2, y2)][-5:]
                    }
                    used_track_ids.add(track_id)
                    matched = True
                    break
            
            if not matched:
                new_track_id = self.next_track_id
                updated_boxes[new_track_id] = {
                    'label': label,
                    'conf': conf,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'age': 0,
                    'history': [(x1, y1, x2, y2)]
                }
                used_track_ids.add(new_track_id)
                self.next_track_id += 1
        
        # Handle boxes yang tidak dideteksi lagi
        for track_id, data in self.tracked_boxes.items():
            if track_id not in used_track_ids:
                data['age'] += 1
                if data['age'] < self.box_timeout:
                    updated_boxes[track_id] = data
        
        return updated_boxes
    
    def detect(self, frame):
        start_time = time.time()
        self.frame_count += 1
        
        current_detections = []
        cat_detected = False
        
        # RUN DETECTION (optimized untuk kecepatan)
        if self.frame_count % 2 == 0:  # Deteksi setiap 2 frame untuk performa
            try:
                results = self.model(
                    frame,
                    conf=0.35,      # Confidence untuk kucing
                    imgsz=320,      # Resolusi lebih kecil untuk kecepatan
                    device="cpu",
                    verbose=False,
                    half=False,
                    max_det=3       # Maks 3 deteksi
                )
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        # HANYA PROSES KUCING
                        if label == "cat" and conf >= 0.35:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            current_detections.append((label, conf, x1, y1, x2, y2))
                            cat_detected = True
                            
                            # Hanya print jika confidence tinggi
                            if conf > 0.5:
                                print(f"🐱 KUCING TERDETEKSI! Confidence: {conf:.2f}")
                
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
        
        # Update tracking
        self.tracked_boxes = self._assign_track_ids(current_detections)
        
        # MQTT LOGIC - HANYA UNTUK KUCING
        now = time.time()
        cat_count = sum(1 for data in self.tracked_boxes.values() if data['label'] == 'cat' and data['age'] == 0)
        
        if cat_count > 0:
            self.last_cat_detection = now
            
            # Kirim MQTT jika cooldown selesai
            if now - self.last_send > 5:  # Cooldown 5 detik
                print(f"🚀 {cat_count} KUCING → KIRIM MQTT 'CAT'")
                send_feed("CAT")
                self.last_send = now
        
        # DRAW SMOOTH BOUNDING BOXES
        for track_id, data in self.tracked_boxes.items():
            if data['label'] == 'cat' and data['age'] < 3:  # Hanya gambar yang masih fresh
                x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
                conf = data['conf']
                
                # Gambar bounding box dengan efek smooth
                thickness = 2 + int(conf * 2)  # Lebih tebal untuk confidence tinggi
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.cat_color, thickness)
                
                # Label dengan shadow effect untuk readability
                label_text = f"CAT {conf:.2f}"
                # Shadow
                cv2.putText(frame, label_text, (x1+1, y1-9), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Text utama
                cv2.putText(frame, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cat_color, 1)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # Update FPS display setiap 2 detik
        if now - self.last_fps_update > 2:
            avg_time = np.mean(self.processing_times) if self.processing_times else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            # print(f"📊 Processing FPS: {fps:.1f}")
            self.last_fps_update = now
        
        # Display info (smooth dan minimal)
        cat_count_display = sum(1 for data in self.tracked_boxes.values() 
                               if data['label'] == 'cat' and data['age'] < 3)
        
        # Status dengan background untuk readability
        status_text = f"Cats: {cat_count_display}"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (5, 5), (text_size[0] + 15, text_size[1] + 15), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Deteksi status
        if cat_count_display > 0:
            status = "🐱 DETECTED"
            color = (0, 255, 0)
            text_size2 = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (5, 40), (text_size2[0] + 15, text_size2[1] + 50), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Frame counter kecil di pojok
        counter_text = f"F:{self.frame_count}"
        cv2.putText(frame, counter_text, (frame.shape[1] - 70, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame