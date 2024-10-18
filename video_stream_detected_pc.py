from flask import Flask, Response
import cv2
import threading
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 nano model
model = YOLO('yolov8n.pt')  # Make sure you have the 'yolov8n.pt' file or download it from the official source

camera = cv2.VideoCapture('http://192.168.50.4:8090/video_feed') 

def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model(frame)

            # Draw the results on the frame
            annotated_frame = results[0].plot()

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Use yield to stream the video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Webcam Stream</title>
        <style>
            body, html { 
                height: 100%; 
                margin: 0; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                background: black;
            }
            img {
                width: 100vw; 
                height: 100vh; 
            }
        </style>
    </head>
    <body>
        <img src="/video_feed">
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
