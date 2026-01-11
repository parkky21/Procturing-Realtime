import asyncio
import logging
import cv2
import numpy as np
import json
import aiohttp
from livekit import rtc

logger = logging.getLogger(__name__)

# Default API Base URL - ideally passed in constructor or env var
DEFAULT_API_URL = "ws://localhost:8080/ws/proctor" 

class Procturing:
    def __init__(self, participant, room, segment_seconds=60, mongo_service=None, api_url=DEFAULT_API_URL):
        self.participant = participant
        self.room = room
        self.session_id = f"{room.name}_{participant.identity}"
        self.api_url = api_url
        
        self.video_track: rtc.VideoTrack = None
        self.audio_track: rtc.AudioTrack = None
        
        self.is_running = False
        self.tasks = []
        self.session = None # aiohttp session
        
        logger.info(f"Procturing Client initialized for {self.session_id}")

    async def start_if_ready(self):
        if self.video_track and not self.is_running:
            self.is_running = True
            
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Start Loops
            self.tasks.append(asyncio.create_task(self._video_loop()))
            self.tasks.append(asyncio.create_task(self._event_loop()))
            
            # If audio is already preserved
            if self.audio_track:
                self.tasks.append(asyncio.create_task(self._audio_loop()))
                
            logger.info("Procturing Client started.")

    async def enable_audio(self, track: rtc.AudioTrack):
        self.audio_track = track
        if self.is_running:
            # Check if audio loop already running? simpler to just fire it
            self.tasks.append(asyncio.create_task(self._audio_loop()))
            logger.info("Audio forwarding enabled.")

    async def stop(self):
        self.is_running = False
        
        # Cancel tasks
        for t in self.tasks:
            t.cancel()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            
        logger.info("Procturing Client stopped.")

    async def _video_loop(self):
        ws_url = f"{self.api_url}/{self.session_id}/video"
        logger.info(f"Connecting to Video WS: {ws_url}")
        
        async with self.session.ws_connect(ws_url, heartbeat=20) as ws:
            video_stream = rtc.VideoStream(self.video_track)
            
            logger.info("Starting video stream forwarding...")
            async for event in video_stream:
                if not self.is_running:
                    break
                    
                frame: rtc.VideoFrame = event.frame
                
                try:
                    # Encoding to JPEG using standard sync cv2 might block loop slightly.
                    # Convert to ARGB numpy
                    img_arr = frame.to_ndarray(format=rtc.VideoBufferType.ARGB)
                    frame_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                    
                    # Encode to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame_bgr)
                    if ret:
                        await ws.send_bytes(buffer.tobytes())
                        
                except Exception as e:
                    logger.error(f"Error in video forwarding: {e}")
                    # If socket closed, maybe break?
                    if ws.closed:
                        break

    async def _audio_loop(self):
        ws_url = f"{self.api_url}/{self.session_id}/audio"
        logger.info(f"Connecting to Audio WS: {ws_url}")
        
        async with self.session.ws_connect(ws_url) as ws:
            audio_stream = rtc.AudioStream(self.audio_track)
            logger.info("Starting audio stream forwarding...")
            
            async for event in audio_stream:
                if not self.is_running:
                    break
                
                frame: rtc.AudioFrame = event.frame
                try:
                    # Send raw PCM bytes (LiveKit audio frame data)
                    # Note: Server expects what Deepgram expects. 
                    await ws.send_bytes(frame.data.tobytes())
                except Exception as e:
                    logger.error(f"Error in audio forwarding: {e}")
                    if ws.closed:
                        break

    async def _event_loop(self):
        ws_url = f"{self.api_url}/{self.session_id}/events"
        logger.info(f"Connecting to Events WS: {ws_url}")
        
        async with self.session.ws_connect(ws_url) as ws:
            logger.info("Listening for alerts...")
            async for msg in ws:
                if not self.is_running:
                    break
                    
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    alerts = data.get("alerts", [])
                    if alerts:
                        logger.warning(f"ALERT [{self.session_id}]: {alerts}")
                        # HERE: You would integrate with your DB service to log these alerts
                        # if self.mongo_service: ...
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
