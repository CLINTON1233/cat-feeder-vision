import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.camera import Camera
from app.detector import CatDetector
from app.mqtt_client import connect
import app.mqtt_client as mqtt_client 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connect()
camera = Camera()
detector = CatDetector()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame = detector.detect(frame)
        frame = cv2.resize(frame, (480, 270))

        _, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 55]
        )

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )

# STREAM VIDEO
@app.get("/video")
def video_stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# API STATUS (ESP32 / MQTT)
@app.get("/status")
def get_status():
    return {
        "cooldown": mqtt_client.cooldown_remaining
    }

# HALAMAN WEB
@app.get("/")
def index():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cat Detector Camera</title>
        <style>
            body {
                background: #111;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
            }
            img {
                border: 3px solid #00ff88;
                border-radius: 10px;
                margin-top: 20px;
                max-width: 90%;
            }
            #statusText {
                color: #00ff88;
                margin-top: 10px;
                font-size: 18px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>Cat Detector - Live Camera</h1>
        <div id="statusText">Cats: LIVE | ESP32 Status: WAITING...</div>
        <img src="/video" />

        <script>
        async function updateStatus(){
            try{
                const res = await fetch("/status");
                const data = await res.json();
                document.getElementById("statusText").innerText =
                    `Cats: LIVE | ESP32 Status: ${data.cooldown}`;
            }catch(err){
                console.log(err);
            }
        }

        setInterval(updateStatus, 1000);
        </script>
    </body>
    </html>
    """)

# RUN SERVER
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
