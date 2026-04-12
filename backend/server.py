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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

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
logger.info("Loading YOLO model...")
model = YOLO("yolov8n.pt")
logger.info("YOLO model loaded.")

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


# ── Frame processing ───────────────────────────────────────────────────────────
def process_frame(frame_bytes: bytes) -> Dict:
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    results = model(img, classes=0, verbose=False)

    count = 0
    annotated_img = img.copy()

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

    _, encoded_img = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64_image = base64.b64encode(encoded_img).decode("utf-8")

    return {
        "status": status,
        "count": count,
        "image_b64": b64_image,
        "timestamp": time.time()
    }


# ── WebSocket: Camera (Mobile App) ────────────────────────────────────────────
@app.websocket("/ws/camera/{code}")
async def websocket_camera(websocket: WebSocket, code: str):
    code = code.upper()
    session = get_session(code)
    if session is None:
        await websocket.close(code=4404, reason="Invalid or expired session code")
        return

    await websocket.accept()
    session.camera_connections.append(websocket)
    session.touch()
    logger.info(f"[{code}] Camera connected. Total cams: {len(session.camera_connections)}")

    try:
        while True:
            data = await websocket.receive_bytes()
            session.touch()
            try:
                result_payload = process_frame(data)
                payload_str = json.dumps(result_payload)
                dead = []
                for dash_ws in session.dashboard_connections:
                    try:
                        await dash_ws.send_text(payload_str)
                    except Exception:
                        dead.append(dash_ws)
                for d in dead:
                    session.dashboard_connections.remove(d)
            except Exception as e:
                logger.error(f"[{code}] Inference error: {e}")
    except WebSocketDisconnect:
        session.camera_connections.remove(websocket)
        logger.info(f"[{code}] Camera disconnected.")


# ── WebSocket: Dashboard (Web Browser) ────────────────────────────────────────
@app.websocket("/ws/dashboard/{code}")
async def websocket_dashboard(websocket: WebSocket, code: str):
    code = code.upper()
    session = get_session(code)
    if session is None:
        await websocket.close(code=4404, reason="Invalid or expired session code")
        return

    await websocket.accept()
    session.dashboard_connections.append(websocket)
    session.touch()
    logger.info(f"[{code}] Dashboard connected. Total dashboards: {len(session.dashboard_connections)}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        session.dashboard_connections.remove(websocket)
        logger.info(f"[{code}] Dashboard disconnected.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
