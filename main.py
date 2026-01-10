from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import threading
import time
import asyncio
import json
from collections import deque

# Import shared models and processing functions
# Import shared models and processing functions
from eye_tracker import process_eye
from head_pose_estimation import process_head_pose
from person_and_phone import process_person_phone
from ultralytics import YOLO
from mediapipe_handler import MediaPipeHandler

app = FastAPI()

# Global event queue for alerts
# Maxlen 1: We only care about the LATEST frame's alerts. 
# This prevents a queue of 20 old "Looking Left" alerts from playing out over 2 seconds after the user stops.
alert_queue = deque(maxlen=1) 

# Initialize Models
yolo_model = YOLO("yolo11n.pt") 
mediapipe_handler = MediaPipeHandler()
 
# Lock not strictly necessary for append/pop left in CPython due to GIL but good practice for clarity needed? 
# Deque is thread-safe for appends and pops.

# HTML for viewing the stream
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>Proctoring AI Live Stream</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; text-align: center; background: #f0f2f5; margin: 0; padding: 20px; }
                h1 { color: #333; margin-bottom: 10px; }
                .container { display: flex; flex-direction: column; align-items: center; }
                img { border: 5px solid #333; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 90%; }
                
                /* Toast Notifications */
                #toast-container {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1000;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .toast {
                    background-color: #ff4d4f; /* Red for alert */
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    opacity: 0;
                    transform: translateX(50px);
                    animation: slideIn 0.3s forwards, fadeOut 0.5s 4s forwards;
                    font-weight: bold;
                    min-width: 250px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                @keyframes slideIn {
                    to { opacity: 1; transform: translateX(0); }
                }
                @keyframes fadeOut {
                    to { opacity: 0; transform: translateX(50px); pointer-events: none; }
                }
            </style>
        </head>
        <body>
            <div id="toast-container"></div>
            
            <div class="container">
                <h1>Proctoring AI - Live Proctoring Feed</h1>
                <p>Real-time analysis of Eye Gaze, Head Pose, Mouth Opening, and Person/Phone Detection.</p>
                <img src="/video_feed" width="800" />
            </div>

            <script>
                const evtSource = new EventSource("/events");
                
                evtSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    const alerts = data.alerts;
                    
                    if (alerts && alerts.length > 0) {
                        alerts.forEach(alertText => {
                            showToast(alertText);
                        });
                    }
                };
                
                function showToast(message) {
                    const container = document.getElementById('toast-container');
                    
                    // Simple infinite loop prevention: don't show duplicates if last one is same?
                    // For now, raw stream.
                    
                    const toast = document.createElement('div');
                    toast.className = 'toast';
                    toast.innerText = message;
                    
                    // Add close button logic if needed, but auto-dismiss is fine
                    container.appendChild(toast);
                    
                    // Remove from DOM after animation
                    setTimeout(() => {
                        if (toast.parentNode) {
                            toast.parentNode.removeChild(toast);
                        }
                    }, 4500); 
                }
            </script>
        </body>
    </html>
    """

def generate_frames():
    # Attempt to open camera (0 is usually default webcam)
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    # Store last known alerts for object detection to persist them between checks
    last_phone_alerts = []

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        try:
            # Collect all alerts for this frame
            frame_alerts = []
            
            # 1. MediaPipe Face Mesh (Unified Eye & Head Tracking)
            mp_results = mediapipe_handler.process(frame)
            
            if mp_results.multi_face_landmarks:
                for face_landmarks in mp_results.multi_face_landmarks:
                    # Eye Tracking (Iris)
                    frame, eye_alerts = process_eye(frame, face_landmarks)
                    frame_alerts.extend(eye_alerts)

                    # Head Pose (Landmarks)
                    frame, head_alerts = process_head_pose(frame, face_landmarks)
                    frame_alerts.extend(head_alerts)
                    
                    # Mouth detection removed as requested
            
            # Optimize: Run YOLO every 30 frames (~1 sec)
            if frame_count % 20 == 0:
                # Actually, let's just run detection.
                processed_frame, phone_alerts = process_person_phone(frame, yolo_model)
                frame = processed_frame # Update frame with boxes
                last_phone_alerts = phone_alerts
            else:
                pass
            
            frame_alerts.extend(last_phone_alerts)
            
            # Push unique alerts to global queue for SSE
            if frame_alerts:
                # Deduplicate within frame
                unique_alerts = list(set(frame_alerts))
                # Push to queue
                alert_queue.append(unique_alerts)

            # NOTE: We removed cv2.putText to keep video clean

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
                   
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
async def event_stream():
    async def event_generator():
        while True:
            if alert_queue:
                # Get oldest alerts
                alerts = alert_queue.popleft()
                # Yield SSE format
                yield f"data: {json.dumps({'alerts': alerts})}\n\n"
            
            # Wait a bit to avoid busy loop and allow queue to fill
            # also acts as rate limiting
            await asyncio.sleep(0.5) 

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
