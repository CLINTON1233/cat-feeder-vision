import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from app.camera import Camera
from app.detector import CatDetector

from app.mqtt_client import connect

app = FastAPI()

connect() 
camera = Camera()
detector = CatDetector()


def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        frame = detector.detect(frame)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )


# 🔹 STREAM VIDEO
@app.get("/video")
def video_stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# 🔹 HALAMAN WEB
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
        </style>
    </head>
    <body>
        <h1> Cat Detector - Live Camera</h1>
        <img src="/video" />
    </body>
    </html>
    """)


# 🔹 JALAN LANGSUNG VIA python main.py
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
