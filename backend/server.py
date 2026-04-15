import asyncio
import base64
import json
import logging
import os
import random
import string
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrowdPulse")

app = FastAPI(title="CrowdPulse AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── YOLO model ────────────────────────────────────────────────────────────────
ML_NODE_URL = os.environ.get("ML_NODE_URL")

if ML_NODE_URL:
    logger.info(f"Relay Mode Enabled: Forwarding inference to {ML_NODE_URL}")
    model = None
else:
    logger.info("Local ML Mode Enabled: Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    logger.info("YOLO medium model loaded for high accuracy.")

# ── Session Store ─────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS = 86400  # 24 hours

class Session:
    def __init__(self, code: str):
        self.code = code
        self.created_at = time.time()
        self.last_active = time.time()
        self.camera_connections: List[WebSocket] = []
        self.dashboard_connections: List[WebSocket] = []

    def touch(self):
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS

sessions: Dict[str, Session] = {}


def generate_code(length: int = 6) -> str:
    """Generate a unique uppercase alphanumeric session code."""
    chars = string.ascii_uppercase + string.digits
    while True:
        code = "".join(random.choices(chars, k=length))
        if code not in sessions:
            return code


def get_session(code: str) -> Optional[Session]:
    session = sessions.get(code.upper())
    if session and session.is_expired():
        del sessions[code.upper()]
        return None
    return session


# ── Background cleanup task ────────────────────────────────────────────────────
async def cleanup_expired_sessions():
    while True:
        await asyncio.sleep(3600)  # run every hour
        expired = [c for c, s in sessions.items() if s.is_expired()]
        for code in expired:
            del sessions[code]
            logger.info(f"Session {code} expired and removed.")


@app.on_event("startup")
async def startup():
    asyncio.create_task(cleanup_expired_sessions())


# ── REST Endpoints ─────────────────────────────────────────────────────────────
class SessionResponse(BaseModel):
    code: str
    expires_in_hours: int = 24


@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """Web dashboard calls this to get a fresh 6-char session code."""
    code = generate_code()
    sessions[code] = Session(code)
    logger.info(f"Session created: {code}")
    return SessionResponse(code=code)


@app.get("/session/{code}/exists")
async def session_exists(code: str):
    """Mobile app calls this to validate a code before connecting."""
    session = get_session(code.upper())
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    session.touch()
    return {"valid": True, "code": code.upper()}


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}

@app.delete("/session/{code}")
async def delete_session(code: str):
    """Explicitly terminate a session and drop all connected clients instantly."""
    code = code.upper()
    if code in sessions:
        session = sessions[code]
        # Drop cameras
        for cam_ws in session.camera_connections:
            try:
                await cam_ws.close(code=4404, reason="Session terminated explicitly")
            except Exception:
                pass
        # Drop dashboards
        for dash_ws in session.dashboard_connections:
            try:
                await dash_ws.close(code=4404, reason="Session terminated explicitly")
            except Exception:
                pass
        del sessions[code]
        logger.info(f"Session {code} instantly terminated by user request.")
        return {"status": "deleted"}
    return {"status": "not_found"}

@app.post("/process_inference")
async def process_inference_route(request: Request):
    """Endpoint for the cloud node to send frames to the local ML node."""
    if ML_NODE_URL:
        raise HTTPException(status_code=400, detail="This node is configured as a relay.")
    data = await request.body()
    result = await asyncio.to_thread(process_frame, data)
    result["jpeg_bytes"] = base64.b64encode(result["jpeg_bytes"]).decode("utf-8")
    return result


# ── Frame processing ───────────────────────────────────────────────────────────
# PERF FIX: Pre-resize to 320x320 before YOLO inference.
# Root cause of 55-83 second inference times on Render free tier:
#   YOLO was receiving a full 640x480 image and internally rescaling to its
#   inference size. We previously shrunk this to 320x320 for Render free tier.
#   Since we process locally now, we bump this to 640 to find small/distant targets.
INFERENCE_SIZE = 640

def process_frame(frame_bytes: bytes) -> Dict:
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    # PERF: Resize to inference size — this is the primary speedup
    h, w = img.shape[:2]
    small_img = cv2.resize(img, (INFERENCE_SIZE, INFERENCE_SIZE), interpolation=cv2.INTER_LINEAR)

    results = model(small_img, classes=0, verbose=False, imgsz=INFERENCE_SIZE)

    count = 0
    annotated_img = small_img.copy()

    if len(results) > 0:
        result = results[0]
        count = len(result.boxes)
        annotated_img = result.plot()

    if count <= 5:
        status = "GREEN"
    elif count <= 15:
        status = "YELLOW"
    else:
        status = "RED"

    # Output at 50% quality — smaller payload = faster dashboard delivery
    _, encoded_img = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 50])

    return {
        "status": status,
        "count": count,
        "jpeg_bytes": encoded_img.tobytes(),
        "timestamp": time.time()
    }


# ── WebSocket: Camera (Mobile App) ────────────────────────────────────────────
@app.websocket("/ws/camera/{code}")
async def websocket_camera(websocket: WebSocket, code: str):
    code = code.upper()
    session = get_session(code)
    if session is None:
        sessions[code] = Session(code)
        session = sessions[code]
        logger.info(f"[{code}] Session auto-created from camera device.")

    await websocket.accept()
    session.camera_connections.append(websocket)
    session.touch()
    logger.info(f"[{code}] Camera connected. Total cams: {len(session.camera_connections)}")

    # FIX: Added 'running' flag for clean cooperative shutdown between workers
    processing_state = {
        "latest_frame": None,
        "running": True,
    }

    async def receive_worker():
        """Continuously receives frames from the camera, always keeping only the newest."""
        try:
            while processing_state["running"]:
                data = await websocket.receive_bytes()
                recv_time = time.time()
                session.touch()
                # FIX: Always overwrite with newest frame — old unprocessed frames are discarded
                # This prevents buffer buildup that caused the "stuck then jump" behavior
                processing_state["latest_frame"] = data
                processing_state["latest_frame_time"] = recv_time
                logger.info(f"[{code}] [RECEIVER] Frame arrived from Android and queued. Network/Wait gap: ~{int((time.time() - recv_time)*1000)}ms")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[{code}] Receive worker error: {e}")
        finally:
            # Signal process_worker to stop when camera disconnects
            processing_state["running"] = False

    async def process_worker():
        """
        Processes frames as fast as YOLO allows and forwards to dashboards.

        FIX 1: Removed the blanket asyncio.sleep(0.1) that was adding 100ms
                on top of every inference cycle (YOLO already takes 200-500ms).
        FIX 2: Only sends a frame to dashboards when a NEW frame was just
                processed — no longer re-broadcasts stale frames repeatedly.
        FIX 3: When no frame is queued, yields with a short sleep instead of
                busy-waiting so the event loop stays responsive.
        """
        active_sends = set()
        last_sent_payload = None
        last_sent_jpeg = None
        
        try:
            while processing_state["running"]:
                data = processing_state["latest_frame"]

                if data is None:
                    # No frame queued — yield briefly and check again
                    # FIX: 10ms yield instead of the old 100ms blanket sleep
                    await asyncio.sleep(0.01)
                    continue

                # Claim and clear the frame atomically
                processing_state["latest_frame"] = None
                frame_born_time = processing_state.get("latest_frame_time", time.time())
                queue_wait_ms = int((time.time() - frame_born_time) * 1000)

                try:
                    logger.info(f"[{code}] [PROCESS] Yanked frame from queue. It waited {queue_wait_ms}ms. Starting YOLO inference...")
                    inf_start = time.time()
                    
                    if ML_NODE_URL:
                        # Relay to external endpoint
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            resp = await client.post(f"{ML_NODE_URL.rstrip('/')}/process_inference", content=data)
                            resp.raise_for_status()
                            result = resp.json()
                            result["jpeg_bytes"] = base64.b64decode(result["jpeg_bytes"])
                    else:
                        # Offload CPU-heavy YOLO inference to a thread pool
                        # so the async event loop isn't blocked during inference
                        result = await asyncio.to_thread(process_frame, data)

                    inf_time_ms = int((time.time() - inf_start) * 1000)
                    logger.info(f"[{code}] [PROCESS] YOLO Inference proxy loop completed in {inf_time_ms}ms. Found {result['count']} people.")

                    last_sent_jpeg = result["jpeg_bytes"]
                    del result["jpeg_bytes"]
                    last_sent_payload = json.dumps(result)

                except Exception as e:
                    logger.error(f"[{code}] Inference error: {e}")
                    continue  # Skip sending on error, wait for next frame

                # FIX: Send ONLY when a new frame was just processed
                if last_sent_jpeg is not None:
                    async def send_to_dash(ws, payload, jpeg):
                        try:
                            if payload is not None:
                                await ws.send_text(payload)
                            await ws.send_bytes(jpeg)
                        except Exception:
                            if ws in session.dashboard_connections:
                                session.dashboard_connections.remove(ws)
                        finally:
                            active_sends.discard(ws)
                                
                    for dash_ws in session.dashboard_connections.copy():
                        if dash_ws in active_sends:
                            # The web app is still downloading the previous frame. 
                            # Drop this frame to stay in perfect real-time.
                            logger.info(f"[{code}] [NETWORK] 🚨 DROPPED frame for dashboard (still sending previous frame). Downlink is slow.")
                            continue
                            
                        active_sends.add(dash_ws)
                        logger.info(f"[{code}] [NETWORK] 🚀 Dispatching new frame block to dashboard...")
                        asyncio.create_task(send_to_dash(dash_ws, last_sent_payload, last_sent_jpeg))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{code}] Process worker error: {e}")

    receive_task = asyncio.create_task(receive_worker())
    process_task = asyncio.create_task(process_worker())

    # FIX: Wait for receive_task to finish first (camera disconnected),
    # then cleanly stop and await process_task.
    # Old code called process_task.cancel() AFTER gather() returned,
    # which meant the cancel raced with already-finished coroutines.
    await receive_task
    processing_state["running"] = False
    process_task.cancel()
    try:
        await process_task
    except asyncio.CancelledError:
        pass

    if websocket in session.camera_connections:
        session.camera_connections.remove(websocket)
    logger.info(f"[{code}] Camera disconnected.")


# ── WebSocket: Dashboard (Web Browser) ────────────────────────────────────────
@app.websocket("/ws/dashboard/{code}")
async def websocket_dashboard(websocket: WebSocket, code: str):
    code = code.upper()
    session = get_session(code)
    if session is None:
        sessions[code] = Session(code)
        session = sessions[code]
        logger.info(f"[{code}] Session auto-created from dashboard.")

    await websocket.accept()
    session.dashboard_connections.append(websocket)
    session.touch()
    logger.info(f"[{code}] Dashboard connected. Total dashboards: {len(session.dashboard_connections)}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in session.dashboard_connections:
            session.dashboard_connections.remove(websocket)
        logger.info(f"[{code}] Dashboard disconnected.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)