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
from app.services.vision.eye_tracker import process_eye
from app.services.vision.head_pose_estimation import process_head_pose
from app.services.vision.person_and_phone import process_person_phone
from ultralytics import YOLO
from app.services.vision.mediapipe_handler import MediaPipeHandler
from app.services.audio_handler import AudioHandler
from contextlib import asynccontextmanager
from fastapi import WebSocket, WebSocketDisconnect
from app.services.session_manager import SessionManager

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from app.services.violation_tracker import ViolationTracker

# Initialize AudioHandler globally (For local mode only)
audio_handler = AudioHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    audio_handler.start(use_microphone=False)
    yield
    # Shutdown
    audio_handler.stop()

app = FastAPI(lifespan=lifespan)

# Global event queue for alerts
# Maxlen 1: We only care about the LATEST frame's alerts. 
# This prevents a queue of 20 old "Looking Left" alerts from playing out over 2 seconds after the user stops.
alert_queue = deque(maxlen=1) 

# Initialize Models
yolo_model = YOLO("models/yolo11n.pt") 
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
    
    # Start microphone if not already running (Local Mode)
    audio_handler.start_microphone()
    
    frame_count = 0
    # Store last known alerts to persist them when skipping frames
    last_mp_alerts = []
    last_yolo_alerts = []
    
    # Procturing Trackers
    # Cooldown: 30 seconds
    head_tracker = ViolationTracker(tolerance_seconds=2.0, cooldown_seconds=30.0)
    eye_tracker = ViolationTracker(tolerance_seconds=2.0, cooldown_seconds=30.0)
    # No person tracker needed for local debug usually, or added if needed?
    # Adding for consistency.
    person_tracker = ViolationTracker(tolerance_seconds=1.0, cooldown_seconds=30.0)
    phone_tracker = ViolationTracker(tolerance_seconds=0.5, cooldown_seconds=30.0)

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
            
            # 1. MediaPipe Face Mesh (Every 3 frames)
            if frame_count % 3 == 0:
                mp_results = mediapipe_handler.process(frame)
                
                current_mp_alerts = []
                
                eye_bad = False
                head_bad = False
                
                eye_details = []
                head_details = []
                
                if mp_results.face_landmarks:
                    for face_landmarks in mp_results.face_landmarks:
                        # Eye Tracking (Iris)
                        frame, eye_raw = process_eye(frame, face_landmarks)
                        if eye_raw:
                            eye_bad = True
                            eye_details.extend(eye_raw) # e.g. ["Looking Right"]

                        # Head Pose (Landmarks)
                        frame, head_raw = process_head_pose(frame, face_landmarks)
                        if head_raw:
                            head_bad = True
                            head_details.extend(head_raw) # e.g. ["Head Down"]
                
                # Update Trackers
                # Show warnings locally? For now let's only show VIOLATIONS to match production alerts
                # Or maybe show warnings in yellow? Stick to violations for simplicity.
                eye_status = eye_tracker.update(eye_bad)
                if eye_status == "VIOLATION":
                     if eye_details:
                         current_mp_alerts.extend(list(set(eye_details)))
                     else:
                         current_mp_alerts.append("Eye Gaze Violation")
                
                head_status = head_tracker.update(head_bad)
                if head_status == "VIOLATION":
                    if head_details:
                        current_mp_alerts.extend(list(set(head_details)))
                    else:
                        current_mp_alerts.append("Head Pose Violation")
                    
                last_mp_alerts = current_mp_alerts
            
            frame_alerts.extend(last_mp_alerts)
            
            # 2. YOLO Object Detection (Every 30 frames)
            if frame_count % 20 == 0:
                processed_frame, phone_alerts = process_person_phone(frame, yolo_model)
                
                # Track Person
                person_bad = any("More than one person" in a for a in phone_alerts)
                person_status = person_tracker.update(person_bad)
                
                # Track Phone
                phone_bad = any(("Phone" in a or "Cell" in a or "Mobile" in a) for a in phone_alerts)
                phone_status = phone_tracker.update(phone_bad)
                
                current_yolo_alerts = []
                
                if person_status == "VIOLATION":
                     current_yolo_alerts.append("Multiple Persons Detected")
                     
                if phone_status == "VIOLATION":
                     current_yolo_alerts.append("Mobile Phone Detected")
                     
                # Only add to frame_alerts ONCE when detected
                frame_alerts.extend(current_yolo_alerts)
                
                # Do NOT persist
                last_yolo_alerts = []
                
            # frame_alerts.extend(last_yolo_alerts) # REMOVED: Caused duplication
            
            # Check for audio alerts
            audio_alerts = audio_handler.get_latest_alerts()
            if audio_alerts:
                frame_alerts.extend(audio_alerts)
            
            # Push unique alerts to global queue for SSE
            if frame_alerts:
                # Deduplicate within frame
                unique_alerts = list(set(frame_alerts))
                # Push to queue
                alert_queue.append(unique_alerts)
                # logger.info(f"[LOCAL] Detections: {unique_alerts}")
                
                # Draw alerts on frame
                y_offset = 50
                for alert in unique_alerts:
                    color = (0, 0, 255) # Red
                    if "Audio" in alert or "speakers" in alert:
                         color = (255, 0, 0) # Blue for audio
                    
                    cv2.putText(frame, str(alert), (30, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    y_offset += 40

            # NOTE: We removed cv2.putText to keep video clean -> Added back for local debug

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
                   
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
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

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- WebSocket Microservice Endpoints ---

session_manager = SessionManager.get_instance()

@app.websocket("/ws/proctor/{session_id}/video")
async def websocket_video(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)
    
    frame_counter = 0
    start_time = time.time()
    
    try:
        while True:
            # Expecting binary JPEG frames
            data = await websocket.receive_bytes()
            
            # FPS Calculation
            frame_counter += 1
            if frame_counter % 30 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_counter / elapsed
                    logger.info(f"Session {session_id} Video FPS: {fps:.2f} (Frames: {frame_counter})")
                    # Reset periodically to keep average relevant to recent performance? 
                    # Or keep cumulative? Cumulative is safer for overall stats, 
                    # but rolling is better for live debugging.
                    # Let's do a rolling window every 30 frames.
                    frame_counter = 0
                    start_time = time.time()
            
            await session.process_video_frame(data)
    except WebSocketDisconnect as e:
        logger.info(f"WS Video Client Disconnected with code: {e.code}")
        # Don't kill session immediately, other sockets might be active.
        # Let some cleanup policy handle it or rely on explicit close.
        pass
    except Exception as e:
        logger.error(f"WS Video Error: {e}")

@app.websocket("/ws/proctor/{session_id}/audio")
async def websocket_audio(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)
    try:
        while True:
            # Expecting binary requests (raw audio)
            data = await websocket.receive_bytes()
            session.process_audio_chunk(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Audio Error: {e}")

@app.websocket("/ws/proctor/{session_id}/events")
async def websocket_events(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_or_create_session(session_id)
    session.attach_event_socket(websocket)
    try:
        while True:
            # Keep connection open. Client might send control messages or just listen.
            # We just wait for disconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        # If event socket disconnects, we might consider the user gone.
        session_manager.remove_session(session_id)
    except Exception as e:
        print(f"WS Events Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
