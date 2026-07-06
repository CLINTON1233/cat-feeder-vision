# Cat Feeder Vision - Smart Cat Detection System

A real-time cat detection system using YOLOv8 object detection, computer vision, and MQTT communication to automatically detect cats and trigger feeding mechanisms through an ESP32 microcontroller.

![Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Usage](#usage)
- [System Components](#system-components)
- [Network Connections](#network-connections)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

**Cat Feeder Vision** is an intelligent pet feeding system that uses computer vision to detect cats and automatically trigger feeders. The system comprises:

- **Raspberry Pi/Linux Computer**: Runs the main Python application with YOLOv8 cat detection
- **USB/Camera Module**: Captures real-time video feed
- **ESP32 Microcontroller**: Controls the feeder mechanism
- **Web Dashboard**: Live video streaming with cat detection visualization
- **MQTT Broker**: Communication bridge between Raspberry Pi and ESP32

### Key Features

✅ Real-time cat detection using YOLOv8 nano model  
✅ Live video streaming with bounding box visualization  
✅ Automatic feeding trigger when cat is detected  
✅ MQTT-based communication with ESP32  
✅ Web dashboard for monitoring  
✅ CORS-enabled API for external integration  
✅ Cooldown mechanism to prevent continuous feeding  
✅ Performance optimized for Raspberry Pi

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RASPBERRY PI / LINUX                  │
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Camera     │      │   FastAPI    │               │
│  │   (USB)      │─────▶│   Server     │               │
│  │              │      │  Port 8000   │               │
│  └──────────────┘      └──────────────┘               │
│                              │                         │
│                              ├─▶ /video (Stream)       │
│                              ├─▶ /status (Status)      │
│                              └─▶ / (Web Dashboard)     │
│                              │                         │
│                        ┌─────▼──────┐                 │
│                        │ YOLO v8n    │                 │
│                        │ Detector    │                 │
│                        └─────┬──────┘                 │
│                              │                         │
│                        ┌─────▼──────┐                 │
│                        │ MQTT Client │                 │
│                        └─────┬──────┘                 │
└─────────────────────────────┼─────────────────────────┘
                              │
                    MQTT BROKER (broker.emqx.io)
                              │
                              │
┌─────────────────────────────▼─────────────────────────┐
│                    ESP32 MICROCONTROLLER               │
│                                                       │
│  ┌──────────────┐      ┌──────────────┐             │
│  │ MQTT Client  │      │  Feeder      │             │
│  │              │─────▶│ Motor/Servo  │             │
│  │              │      │              │             │
│  └──────────────┘      └──────────────┘             │
│                                                       │
│              Cooldown Timer Management               │
└───────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Essential Components

1. **Raspberry Pi (or Linux Computer)**
   - Raspberry Pi 4 (4GB+ RAM recommended) or better
   - Raspberry Pi OS or Ubuntu Linux
   - Power supply (5V/3A minimum)

2. **Camera Module**
   - USB Camera (720p or higher)
   - OR Raspberry Pi Camera Module (CSI/DSI)
   - 30+ FPS capability recommended

3. **ESP32 Microcontroller**
   - ESP32 Development Board
   - WiFi capabilities
   - GPIO pins for controlling feeder motor/servo

4. **Feeder Mechanism**
   - DC Motor or Servo Motor
   - Motor driver (L298N or similar)
   - Power supply for motor

5. **Network**
   - WiFi or Ethernet connectivity
   - MQTT Broker access (cloud or local)

### Optional Components

- Cooling fan for Raspberry Pi (if running continuously)
- External power bank
- Status LED indicators
- Temperature sensor

---

## Installation Guide

### Step 1: Prerequisites

**On Raspberry Pi/Linux:**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies for OpenCV
sudo apt install -y libatlas-base-dev libjasper-dev libharfp0
sudo apt install -y libwebp6 libtiff5 libjasper1 libqtgui4 libqt4-test libhdf5-dev libharfp0
```

### Step 2: Clone or Download Project

```bash
cd ~/
git clone <your-repo-url> cat-feeder-vision
cd cat-feeder-vision
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Dependencies explained:**

- **fastapi**: Web framework for REST API
- **uvicorn**: ASGI web server for FastAPI
- **opencv-python**: Computer vision library for video capture
- **numpy**: Numerical computing library
- **ultralytics**: YOLOv8 framework and models
- **paho-mqtt**: MQTT client for message broker communication

### Step 5: Download YOLOv8 Model

The `yolov8n.pt` file should already be in your project directory. If not, download it:

```bash
# The model will auto-download on first run, or manually download:
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 6: Configure Camera

**For USB Camera:**

```bash
# List available cameras
ls /dev/video*

# Test camera with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture('/dev/video0'); print(cap.isOpened())"
```

**For Raspberry Pi Camera:**

```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to Interface Options → Camera → Enable
```

### Step 7: Verify MQTT Connectivity

```bash
# Test MQTT connection
python3 -c "from app.mqtt_client import connect; connect()"
```

---

## Configuration

### MQTT Broker Settings

Edit `app/mqtt_client.py` to configure your MQTT broker:

```python
BROKER = "broker.emqx.io"      # Change to your broker address
PORT = 1883                     # Default MQTT port (1883 for non-SSL, 8883 for SSL)

TOPIC_FEED = "cat/feeding"      # Topic for feeding commands
TOPIC_STATUS = "cat/status"     # Topic for status updates
```

### Camera Settings

In `app/camera.py`, adjust camera parameters:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # Frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # Frame height
cap.set(cv2.CAP_PROP_FPS, 30)              # Frames per second
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)        # Buffer size (lower = less latency)
```

### YOLOv8 Detector Settings

In `app/detector.py`, adjust detection parameters:

```python
self.cat_color = (0, 255, 0)      # Bounding box color (BGR format)
self.box_timeout = 15              # Tracking timeout in frames
self.last_send = 0                 # Cooldown tracking
```

### FastAPI Settings

In `app/main.py`, modify server settings:

```bash
# Production: modify the uvicorn.run() call at the bottom
uvicorn.run(
    app,
    host="0.0.0.0",      # Listen on all interfaces
    port=8000            # Change port if needed
)
```

---

## Usage

### Starting the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the application
python3 -m app.main

# Or directly
cd ~/cat-feeder-vision
python3 app/main.py
```

### Expected Console Output

```
🔍 Found devices: ['/dev/video0']
🔄 Trying camera: /dev/video0
📷 Camera ACTIVE at /dev/video0
  Resolution: 640x480
  FPS: 30
🎬 Capture thread started

==================================================
🐱 CAT DETECTOR INITIALIZED
==================================================
✅ Detector ready - Only cats will be detected
==================================================

🔌 Connecting MQTT...
🟢 MQTT CONNECTED (RASPBERRY)
📥 SUBSCRIBED TO cat/status
🚀 MQTT LOOP STARTED, RASPBERRY ONLINE

INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Accessing the Web Dashboard

1. Open your browser
2. Navigate to: `http://<raspberry-pi-ip>:8000`
3. You should see:
   - Live video stream with cat detection bounding boxes
   - Real-time status information
   - ESP32 cooldown status

### Running in Background

```bash
# Using nohup
nohup python3 app/main.py > cat_feeder.log 2>&1 &

# Using systemd (recommended for permanent setup)
# Create /etc/systemd/system/cat-feeder.service
sudo systemctl start cat-feeder
sudo systemctl enable cat-feeder  # Auto-start on boot
```

---

## System Components

### 1. Camera Module (`app/camera.py`)

**Functionality:**

- Automatically detects available video devices (`/dev/video*`)
- Initializes camera with optimized settings
- Runs continuous frame capture in separate thread
- Implements queue-based frame buffering for performance
- Auto-configures resolution, FPS, and codec

**Key Methods:**

```python
Camera()          # Initialize and find camera
get_frame()       # Get latest captured frame
```

**Performance Optimization:**

- Multi-threaded capture prevents frame drops
- Small queue (maxsize=2) reduces latency
- MJPEG codec improves performance
- 30 FPS, 640x480 resolution for balance

---

### 2. YOLOv8 Detector (`app/detector.py`)

**Functionality:**

- Uses YOLOv8 nano model for real-time object detection
- Detects cats in video frames
- Draws bounding boxes around detected cats
- Tracks multiple cats across frames
- Implements smoothing for stable bounding boxes

**Key Features:**

- **IOU Matching**: Matches detections across frames using Intersection over Union
- **Track ID Assignment**: Assigns unique IDs to each tracked cat
- **Box Smoothing**: Moving average filter for stable bounding boxes
- **Timeout Management**: Removes old tracks after 15 frames without detection
- **Performance Tracking**: Logs FPS and processing times

**Detection Workflow:**

```
Frame Input
    ↓
YOLOv8 Inference
    ↓
Filter for Cats Only
    ↓
Track Assignment (IOU matching)
    ↓
Box Smoothing
    ↓
Draw Bounding Boxes
    ↓
Send MQTT if New Detection
    ↓
Frame Output
```

---

### 3. MQTT Client (`app/mqtt_client.py`)

**Functionality:**

- Connects to MQTT broker (broker.emqx.io)
- Subscribes to ESP32 status messages
- Publishes cat detection events
- Manages cooldown status from ESP32
- Handles connection/disconnection events

**Topics:**

- **`cat/feeding`**: Publish when cat is detected (triggers feeder)
- **`cat/status`**: Subscribe to ESP32 status (cooldown time, motor state, etc.)

**Connection States:**

- `MQTT CONNECTED`: Successfully connected
- `MQTT FAILED`: Connection error
- `RASPBERRY ONLINE`: Connection established
- `{SOURCE} DETECTED BY CAMERA`: Cat detection notification

---

### 4. FastAPI Server (`app/main.py`)

**Endpoints:**

| Endpoint  | Method | Description                   |
| --------- | ------ | ----------------------------- |
| `/`       | GET    | Web dashboard with live video |
| `/video`  | GET    | MJPEG video stream            |
| `/status` | GET    | Current system status (JSON)  |

**Response Examples:**

```bash
# GET /status
{
  "cooldown": "MQTT CONNECTED"
}

# GET /video
# Returns MJPEG stream (for <img> tag)
```

**Middleware:**

- CORS enabled for all origins
- Supports external API calls

---

## Network Connections

### Data Flow Architecture

```
┌────────────────────────────────────────────────────────┐
│ STEP 1: VIDEO CAPTURE                                  │
│ Camera → Frame Buffer → Application Memory             │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│ STEP 2: OBJECT DETECTION                               │
│ Frame → YOLOv8 Model → Detections (Bounding Boxes)    │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│ STEP 3: WEB STREAMING                                  │
│ Detection Frame → JPEG Encode → HTTP Stream (/video)   │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│ STEP 4: EVENT NOTIFICATION                             │
│ Cat Detected → MQTT Publish (cat/feeding) → Broker     │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│ STEP 5: ESP32 RESPONSE                                 │
│ Broker → MQTT Subscribe → ESP32 → Motor Control       │
└────────────────────────────────────────────────────────┘
```

### Connection Details

#### Raspberry Pi ↔ Camera

- **Type**: USB or CSI/DSI (built-in)
- **Protocol**: Video4Linux (V4L2)
- **Data Rate**: ~15-30 Mbps (depends on resolution/FPS)
- **Detection**: Automatic via `/dev/video*`

#### Raspberry Pi ↔ MQTT Broker

- **Protocol**: TCP/IP (MQTT)
- **Address**: `broker.emqx.io`
- **Port**: `1883` (default)
- **Connection Type**: WiFi or Ethernet
- **Authentication**: None (modify for your setup)

#### MQTT Broker ↔ ESP32

- **Protocol**: TCP/IP (MQTT)
- **Connection Type**: WiFi
- **Topics Subscribed**:
  - `cat/feeding` (listen for feed command)
  - `cat/status` (broadcast status)

#### Raspberry Pi ↔ Web Clients

- **Protocol**: HTTP
- **Port**: `8000` (default)
- **Connection Type**: WiFi or Ethernet
- **URL**: `http://<pi-ip>:8000`

### Network Prerequisites

```
Raspberry Pi Requirements:
├── Internet Connectivity
│   ├── WiFi connection (or Ethernet)
│   ├── Access to broker.emqx.io
│   └── Firewall port 8000 open for web access
│
└── Local Network
    ├── Same WiFi network as client devices
    └── Stable connection (2.4GHz recommended for range)
```

---

## API Documentation

### HTTP Endpoints

#### 1. Get Live Video Stream

**Request:**

```
GET /video
```

**Response:**

- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- Body: Continuous MJPEG stream

**Usage:**

```html
<img src="http://raspberry-pi-ip:8000/video" />
```

**Example Response:**

```
--frame
Content-Type: image/jpeg

[JPEG image binary data]
--frame
Content-Type: image/jpeg

[JPEG image binary data]
...
```

---

#### 2. Get System Status

**Request:**

```
GET /status
```

**Response:**

```json
{
  "cooldown": "MQTT CONNECTED"
}
```

**Status Values:**

- `MQTT CONNECTED`: System ready
- `MQTT FAILED`: Connection error
- `<number> seconds remaining`: Feeder on cooldown
- `RASPBERRY ONLINE`: Successfully connected
- `{OBJECT} DETECTED BY CAMERA`: Object detected

---

#### 3. Web Dashboard

**Request:**

```
GET /
```

**Response:** HTML page with embedded video stream and status

**Features:**

- Live video from `/video` endpoint
- Real-time status display
- Auto-refresh status every 2 seconds
- Responsive design

---

### MQTT Topics

#### Publish: `cat/feeding`

**Trigger:** When cat detected

**Payload:**

```
cat
```

**Example Flow:**

```
1. Camera detects cat
2. Publishes: "cat" to topic "cat/feeding"
3. ESP32 receives message
4. ESP32 activates feeder motor
5. ESP32 publishes status update
```

---

#### Publish: `cat/status`

**Messages:**

```
RASPBERRY ONLINE          # On startup
CAT DETECTED BY CAMERA    # When cat found
{status from ESP32}       # Any message from ESP32
```

---

#### Subscribe: `cat/status`

**Listen for ESP32 updates:**

Example payloads:

```
FEEDER ACTIVATED
COOLDOWN 60 SECONDS
MOTOR ERROR
READY FOR FEEDING
```

---

## Troubleshooting

### Camera Not Detected

**Problem:** `RuntimeError: No working camera detected`

**Solutions:**

```bash
# 1. List available cameras
ls -l /dev/video*

# 2. Check if camera is connected
lsusb  # For USB cameras

# 3. Check camera permissions
sudo usermod -a -G video $USER
# Reboot required

# 4. Test camera directly
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

---

### MQTT Connection Failed

**Problem:** `🔴 MQTT CONNECT FAILED`

**Solutions:**

```bash
# 1. Check internet connection
ping broker.emqx.io

# 2. Verify MQTT broker is reachable
nc -zv broker.emqx.io 1883

# 3. Check firewall settings
sudo ufw allow 1883

# 4. Test MQTT locally
pip install mosquitto-clients
mosquitto_pub -h broker.emqx.io -p 1883 -t "test" -m "hello"
```

---

### High CPU/Memory Usage

**Problem:** Application consuming excessive resources

**Solutions:**

```python
# 1. Reduce frame resolution (app/camera.py):
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 2. Reduce FPS:
cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS

# 3. Use lighter model (app/detector.py):
# Current: YOLO("yolov8n.pt")  # nano
# Try: YOLO("yolov8s.pt")      # small (larger but faster on CPU)
```

---

### Video Stream Lags or Disconnects

**Problem:** Web stream is choppy or drops frequently

**Solutions:**

```python
# 1. Reduce streaming resolution (app/main.py):
frame = cv2.resize(frame, (320, 180))  # Lower streaming resolution

# 2. Increase JPEG compression (app/main.py):
[cv2.IMWRITE_JPEG_QUALITY, 40]  # Lower quality (faster)

# 3. Reduce buffer size (app/camera.py):
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Even smaller buffer
```

---

### ESP32 Not Receiving Commands

**Problem:** MQTT publish not triggering feeder

**Solutions:**

```bash
# 1. Monitor MQTT messages
mosquitto_sub -h broker.emqx.io -p 1883 -t "cat/#"

# 2. Verify topic names match
# Check TOPIC_FEED in app/mqtt_client.py
# Check topic subscribed in ESP32 code

# 3. Check ESP32 connection
# Verify ESP32 also connects to broker.emqx.io

# 4. Test publish manually
mosquitto_pub -h broker.emqx.io -p 1883 -t "cat/feeding" -m "cat"
```

---

### Model Download Issues

**Problem:** `Failed to download YOLOv8 model`

**Solutions:**

```bash
# 1. Manual download
cd ~/cat-feeder-vision
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# 2. Set cache directory
mkdir -p ~/.config/Ultralytics
export YOLOv8_CACHE=~/.config/Ultralytics

# 3. Pre-download in virtual environment
source venv/bin/activate
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print("Model downloaded successfully")
EOF
```

---

## Performance Optimization Tips

### For Raspberry Pi 4 (4GB RAM)

```python
# Reduce detection confidence threshold
results = self.model.predict(frame, conf=0.5)  # Default: 0.25

# Skip every nth frame for detection (run detection on every 2nd frame)
if self.frame_count % 2 == 0:
    results = self.model.predict(frame)

# Use FP16 precision (faster on GPU)
results = self.model.predict(frame, half=True)
```

### Network Optimization

```python
# Reduce streaming resolution and quality for lower bandwidth
frame = cv2.resize(frame, (320, 180))  # 16:9 aspect ratio
[cv2.IMWRITE_JPEG_QUALITY, 35]         # 35% quality
```

### Monitor Resources

```bash
# Check CPU/Memory usage
top -p $(pgrep -f "python3 app/main.py")

# Check disk usage
df -h

# Check temperature
vcgencmd measure_temp  # Raspberry Pi only
```

---

## Advanced Configuration

### Using Local MQTT Broker

Replace `broker.emqx.io` with your local broker:

```python
# app/mqtt_client.py
BROKER = "192.168.1.100"  # Your local MQTT broker IP
PORT = 1883
```

### Enable MQTT Authentication

```python
# app/mqtt_client.py
def on_connect(client, userdata, flags, rc):
    client.username_pw_set("username", "password")
```

### SSL/TLS Connection

```python
# For SSL connections on port 8883
PORT = 8883
client.tls_set(
    ca_certs="/path/to/ca.crt",
    certfile="/path/to/client.crt",
    keyfile="/path/to/client.key"
)
```

---

## Systemd Service Setup

Create `/etc/systemd/system/cat-feeder.service`:

```ini
[Unit]
Description=Cat Feeder Vision System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/cat-feeder-vision
Environment="PATH=/home/pi/cat-feeder-vision/venv/bin"
ExecStart=/home/pi/cat-feeder-vision/venv/bin/python3 app/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cat-feeder
sudo systemctl start cat-feeder
sudo systemctl status cat-feeder
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Support and Contribution

For issues, questions, or improvements, please create an issue or pull request in the repository.

**Last Updated:** 2024  
**Version:** 1.0  
**Maintained by:** Your Name
