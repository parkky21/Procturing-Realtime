import asyncio
import logging
import cv2
import numpy as np
import threading
from collections import deque
from typing import Dict, List, Optional
from fastapi import WebSocket
from ultralytics import YOLO

from .audio_handler import AudioHandler
from .vision.mediapipe_handler import MediaPipeHandler
from .vision.eye_tracker import process_eye
from .vision.head_pose_estimation import process_head_pose
from .vision.person_and_phone import process_person_phone

logger = logging.getLogger(__name__)

# Singleton YOLO model to save memory
# Initialize carefully or lazily if needed. 
try:
    GLOBAL_YOLO = YOLO("models/yolo11n.pt")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    GLOBAL_YOLO = None

class ProctoringSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_active = True
        
        # Audio Handler (Deepgram)
        self.audio_handler = AudioHandler()
        # Start in stream mode (no mic)
        self.audio_handler.start(use_microphone=False)
        
        # Vision Handler (MediaPipe is lightweight, ideally one per session to avoid state concurrency issues)
        self.mp_handler = MediaPipeHandler()
        
        # Throttling Configuration
        self.frame_count = 0
        self.mp_interval = 3      # Run every 3 frames
        self.yolo_interval = 30   # Run every 30 frames
        
        # State persistence for throttling
        self.last_mp_alerts: List[str] = []
        self.last_yolo_alerts: List[str] = []
        
        # Alert Queue (broadcast to event websocket)
        self.event_socket: Optional[WebSocket] = None
        
    async def process_video_frame(self, frame_bytes: bytes):
        """
        Process a single video frame (JPEG bytes).
        Returns list of alerts found in this frame.
        """
        if not self.is_active:
            return

        try:
            # Decode JPEG
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return

            frame_alerts = []
            
            # 1. MediaPipe
            if self.frame_count % self.mp_interval == 0:
                mp_results = self.mp_handler.process(frame)
                current_mp_alerts = []
                if mp_results.face_landmarks:
                    for face_landmarks in mp_results.face_landmarks:
                        _, eye_alerts = process_eye(frame, face_landmarks)
                        current_mp_alerts.extend(eye_alerts)
                        
                        _, head_alerts = process_head_pose(frame, face_landmarks)
                        current_mp_alerts.extend(head_alerts)
                self.last_mp_alerts = current_mp_alerts
            
            frame_alerts.extend(self.last_mp_alerts)
            
            # 2. YOLO
            if GLOBAL_YOLO and (self.frame_count % self.yolo_interval == 0):
                # YOLO handling
                _, phone_alerts = process_person_phone(frame, GLOBAL_YOLO)
                self.last_yolo_alerts = phone_alerts
            
            frame_alerts.extend(self.last_yolo_alerts)
            
            # 3. Audio Alerts (Poll from AudioHandler)
            audio_alerts = self.audio_handler.get_latest_alerts()
            if audio_alerts:
                frame_alerts.extend(audio_alerts)

            self.frame_count += 1
            
            # Broadcast if alerts exist
            if frame_alerts:
                unique_alerts = list(set(frame_alerts))
                # print(f"[{self.session_id}] Detections: {unique_alerts}")
                await self.broadcast_alerts(unique_alerts)
                
        except Exception as e:
            logger.error(f"Error processing video frame for {self.session_id}: {e}")

    def process_audio_chunk(self, audio_bytes: bytes):
        """
        Feed raw audio bytes to Deepgram.
        """
        if self.is_active:
            self.audio_handler.process_audio_chunk(audio_bytes)

    async def broadcast_alerts(self, alerts: List[str]):
        if self.event_socket:
            try:
                import json
                await self.event_socket.send_text(json.dumps({"alerts": alerts}))
            except Exception as e:
                logger.error(f"Failed to send alerts to socket: {e}")

    def attach_event_socket(self, ws: WebSocket):
        self.event_socket = ws

    def close(self):
        self.is_active = False
        self.audio_handler.stop()
        if self.event_socket:
             # Socket cleanup is usually handled by the endpoint handler
             self.event_socket = None
        logger.info(f"Session {self.session_id} closed.")


class SessionManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.sessions: Dict[str, ProctoringSession] = {}

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = SessionManager()
        return cls._instance

    def get_or_create_session(self, session_id: str) -> ProctoringSession:
        if session_id not in self.sessions:
            logger.info(f"Creating new session: {session_id}")
            self.sessions[session_id] = ProctoringSession(session_id)
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[ProctoringSession]:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].close()
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")
