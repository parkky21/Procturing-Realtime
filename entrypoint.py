import asyncio
import logging
from typing import Dict, Optional
from app.helpers.agent_builder import build_llm_instance, build_stt_instance, build_tts_instance
from app.helpers.evaluation_helper import write_transcript_and_evaluate
from app.schemas import Agent
from app.core.single_agent import InterviewAgent
import json
import os
from livekit.plugins import noise_cancellation
from livekit import agents, rtc
from livekit.agents import (
    AgentSession,
    room_io
)
from app.services.mongoDB_service import get_global_mongo_service
from app.services.procturing import Procturing
from app.services.egress_recording_service import EgressRecordingService
from app.services.lemonfox_voice_service import get_voice_by_round_robin
from app.schemas.recording_schemas import Recording

livekit_agents_logger = logging.getLogger("livekit.agents")
livekit_agents_logger.setLevel(logging.WARNING)  # Only show WARNING and above

logger = logging.getLogger(__name__)

# Use global singleton for LiveKit agent context
mongo = get_global_mongo_service(db_name="algojobs")

async def entrypoint(ctx: agents.JobContext):

    metadata = json.loads(ctx.job.metadata)
    original_prompt = metadata.get("prompt", "You are an AI assistant helping with interviews.")
    agent_id = metadata.get("agent_id", "unknown_agent")
    candidate_id = metadata.get("candidate_id")  # Optional: for evaluation
    job_id = metadata.get("job_id")  # Optional: for evaluation
    proctoring_enabled = metadata.get("proctoring_enabled", False)  # Default to True if not specified
    recording_enabled = metadata.get("recording_enabled", False)  # Default to True if not specified
    interview_duration_minutes = metadata.get("interview_duration_minutes")  # Optional: for auto-transfer
    closure_prompt = metadata.get("closure_prompt","")
    owner_id = metadata.get("owner_id")
    
    # Try to get agent config from metadata first (to avoid DB call)
    agent_config_dict = metadata.get("agent_config")
    if agent_config_dict:
        logger.info(f"[ENTRY] Using agent config from metadata for agent_id: {agent_id}")
        try:
            # Convert dict from metadata to Pydantic model
            agent = Agent.model_validate(agent_config_dict)
            agent_config = agent.agentConfig
            if agent_config is None:
                raise ValueError(f"Agent {agent_id} has no agentConfig in metadata")
        except Exception as e:
            logger.warning(f"[ENTRY] Failed to parse agent config from metadata: {e}. Falling back to DB lookup.")
            agent_config_dict = None
    
    # Fallback to DB lookup if agent_config not in metadata (backward compatibility)
    if not agent_config_dict:
        logger.info(f"[ENTRY] Agent config not in metadata, fetching from database for agent_id: {agent_id}")
        agent_doc = mongo.get_agent_config_by_id(agent_id)
        if agent_doc is None:
            error_msg = f"Agent configuration not found for agent_id: {agent_id}. Please verify the agent exists in the database."
            logger.error(f"[ENTRY] {error_msg}")
            raise ValueError(error_msg)
        
        # Convert dict to Pydantic model - now you can use dot notation!
        agent = Agent.model_validate(agent_doc)
        
        # Access using dot notation - agentConfig is already validated as AgentConfig (or None)
        agent_config = agent.agentConfig
        if agent_config is None:
            raise ValueError(f"Agent {agent_id} has no agentConfig")
    if agent_config.delay_mins:
        logger.info(f"[DELAY] delay_mins found in database, using {agent_config.delay_mins}")
        delay_mins = agent_config.delay_mins
    else:
        logger.info(f"[DELAY] delay_mins not found in database, using default {agent_config.delay_mins}")
        delay_mins = 3

    llm = build_llm_instance(
        agent_config.llm.provider, 
        agent_config.llm.model, 
        agent_config.llm.api_key, 
        agent_config.llm.temperature,
        agent_config.llm.max_completion_tokens,
        agent_config.llm.reasoning_effort
        )
    stt = build_stt_instance(
        agent_config.stt.provider,
        agent_config.stt.model,
        agent_config.stt.language,
        agent_config.stt.api_key
    )
    
    # Check if TTS provider is lemonfox and apply round-robin voice selection
    tts_voice = agent_config.tts.voice
    prompt_name = None
    if agent_config.tts.provider == "lemonfox":
        voice_name, prompt_name = get_voice_by_round_robin()
        tts_voice = voice_name
        logger.info(f"[LEMONFOX] Selected voice: {voice_name} with prompt name: {prompt_name}")
    
    tts = build_tts_instance(
        provider=agent_config.tts.provider, 
        model=agent_config.tts.model, 
        sample_rate=8000, 
        language=agent_config.tts.language,
        voice=tts_voice, 
        credentials_info=agent_config.tts.api_key
    )

    await ctx.connect()

    # Initialize recording service if enabled
    recording_service: Optional[EgressRecordingService] = None
    active_recording: Optional[Recording] = None
    
    if recording_enabled:
        try:
            recording_service = EgressRecordingService(mongo_service=mongo)
            interview_id_str = str(ctx.job.id) if hasattr(ctx.job, 'id') and ctx.job.id else None
            room_name = ctx.room.name if ctx.room else None
            
            if room_name:
                # Start room composite recording
                active_recording = await recording_service.start_room_recording(
                    room_name=room_name,
                    interview_id=interview_id_str,
                    candidate_id=candidate_id,
                    job_id=job_id,
                )
                if active_recording:
                    logger.info(f"[RECORDING] Started recording for room {room_name}, egress_id={active_recording.egress_id}")
                else:
                    logger.warning(f"[RECORDING] Failed to start recording for room {room_name}")
            else:
                logger.warning("[RECORDING] Room name not available, skipping recording")
        except Exception as e:
            logger.exception(f"[RECORDING] Error initializing recording service: {e}")
            recording_service = None
    else:
        logger.info("[RECORDING] Recording disabled for this interview")

    # Initialize proctoring only if enabled
    proctoring_instances: Dict[str, Procturing] = {}
    
    if proctoring_enabled:
        segment_seconds = int(os.getenv("PROCTORING_SEGMENT_SECONDS", "60"))
        logger.info(f"[ENTRY] Interview agent started with proctoring enabled. segment_seconds={segment_seconds}")
        
        def get_id(p: rtc.RemoteParticipant) -> str:
            return p.identity or p.sid

        async def ensure_proctoring(p: rtc.RemoteParticipant) -> Procturing:
            pid = get_id(p)
            if pid not in proctoring_instances:
                proctoring_instances[pid] = Procturing(
                    participant=p,
                    room=ctx.room,
                    segment_seconds=segment_seconds,
                    mongo_service=mongo,
                )
                logger.info(f"[PROCTORING] Created proctoring instance for participant={pid}")
            return proctoring_instances[pid]

        def on_participant_connected(p: rtc.RemoteParticipant) -> None:
            logger.info(f"[ROOM] participant_connected: {get_id(p)}")
            # Attempt to attach existing publications
            for pub in p.track_publications.values():
                _maybe_attach(pub, p)

        def on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
            logger.info(f"[ROOM] participant_disconnected: {get_id(p)}")
            pid = get_id(p)
            proc = proctoring_instances.pop(pid, None)
            if proc:
                asyncio.create_task(proc.stop())

        def on_track_subscribed(track: rtc.Track, pub: rtc.TrackPublication, p: rtc.RemoteParticipant) -> None:
            logger.info(f"[ROOM] track_subscribed: kind={pub.kind} source={pub.source} participant={get_id(p)}")
            _maybe_attach(pub, p)

        def _maybe_attach(pub: rtc.TrackPublication, p: rtc.RemoteParticipant) -> None:
            """
            Attach video/audio tracks to proctoring instance.
            Video is required, audio is optional.
            """
            # Video is required, audio is optional
            if pub.kind == rtc.TrackKind.KIND_VIDEO and pub.source == rtc.TrackSource.SOURCE_CAMERA and pub.track:
                async def set_video() -> None:
                    proc = await ensure_proctoring(p)
                    proc.video_track = pub.track  # type: ignore
                    # If audio track already exists, attach it now
                    if proc.audio_track is not None:
                        logger.info(f"[PROCTORING] Video arrived, audio already exists, enabling audio...")
                        await proc.enable_audio(proc.audio_track)
                    await proc.start_if_ready()

                asyncio.create_task(set_video())
            elif pub.kind == rtc.TrackKind.KIND_AUDIO and pub.source == rtc.TrackSource.SOURCE_MICROPHONE and pub.track:
                async def set_audio() -> None:
                    proc = await ensure_proctoring(p)
                    logger.info(f"[PROCTORING] Audio track received for {get_id(p)} - track: {pub.track.sid}")
                    if proc.video_track is not None:
                        logger.info(f"[PROCTORING] Video track exists, calling enable_audio()...")
                        await proc.enable_audio(pub.track)  # type: ignore
                        logger.info(f"[PROCTORING] enable_audio() completed")
                    else:
                        logger.info(f"[PROCTORING] Audio arrived before video for {get_id(p)}; waiting for video")
                        # Store audio track for later
                        proc.audio_track = pub.track  # type: ignore

                asyncio.create_task(set_audio())

        # Wire room events for proctoring
        ctx.room.on("participant_connected", on_participant_connected)
        ctx.room.on("participant_disconnected", on_participant_disconnected)
        ctx.room.on("track_subscribed", on_track_subscribed)

        # Attach existing remote participants
        for p in ctx.room.remote_participants.values():
            on_participant_connected(p)
    else:
        logger.info("[ENTRY] Interview agent started with proctoring disabled")

    session = AgentSession(
        stt=stt, 
        llm=llm, 
        tts=tts,
        turn_detection="vad",
        vad=ctx.proc.userdata["vad"]
        )

    # Create shutdown callbacks
    async def shutdown_callback():
        """Handle shutdown: stop recording and evaluate."""
        # Stop recording if active
        if recording_service and active_recording:
            try:
                logger.info(f"[RECORDING] Stopping recording {active_recording.egress_id}")
                stopped_recording = await recording_service.stop_recording(active_recording.egress_id)
                if stopped_recording:
                    logger.info(f"[RECORDING] Recording stopped successfully. Status: {stopped_recording.status}")
                else:
                    logger.warning(f"[RECORDING] Failed to stop recording {active_recording.egress_id}")
            except Exception as e:
                logger.exception(f"[RECORDING] Error stopping recording: {e}")
            finally:
                # Close recording service
                try:
                    await recording_service.close()
                except Exception as e:
                    logger.exception(f"[RECORDING] Error closing recording service: {e}")
        
        # Evaluate candidate
        # Check if this is a mock interview (proctoring disabled)
        is_mock_interview = not proctoring_enabled
        
        await write_transcript_and_evaluate(
            session=session,
            candidate_id=candidate_id,
            job_id=job_id,
            mongo_service=mongo,
            room_name=ctx.room.name if ctx.room else None,
            interview_id=str(ctx.job.id) if hasattr(ctx.job, 'id') and ctx.job.id else None,
            timeout=120.0,
            owner_id=owner_id,
            is_mock_interview=is_mock_interview
        )

    ctx.add_shutdown_callback(shutdown_callback)

    # Update prompt with name if using lemonfox
    prompt = original_prompt
    if agent_config.tts.provider == "lemonfox" and prompt_name:
        prompt = f"Your name is {prompt_name}.\n{original_prompt}"
        logger.info(f"[LEMONFOX] Updated prompt with name: {prompt_name}")

    agent = InterviewAgent(
        prompt=prompt, 
        closure_prompt=closure_prompt, 
        interview_duration_minutes=interview_duration_minutes,
        delay_mins=delay_mins
    )

    await session.start(
        agent=agent, 
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )