# engine_service.py

import cv2
import time
import numpy as np
import threading
import json
import os
import base64
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Important: Copy the classes from the other files directly into here ---
# This makes the service self-contained.

class CameraBuffer:
    # ... (Paste the entire CameraBuffer class here, unchanged)
    def __init__(self, src, resolution, buffer_size):
        self.src = src; self.resolution = resolution; self.buffer_size = buffer_size
        self.stopped = False; self.stream = None; self.frame_queue = deque(maxlen=buffer_size)
        self.latest_frame = None; self.lock = threading.Lock(); self.connected = False
    def _collector_thread(self):
        while not self.stopped:
            if not self.connected: self.connect(); continue
            grabbed, frame = self.stream.read()
            if not grabbed:
                print("[COLLECTOR] Lost connection. Reconnecting..."); self.stream.release(); self.connected = False; time.sleep(2.0); continue
            with self.lock: self.frame_queue.append(frame); self.latest_frame = frame
    def connect(self):
        print(f"[COLLECTOR] Connecting to: {self.src}..."); self.stream = cv2.VideoCapture(self.src)
        if not self.stream.isOpened(): print("[COLLECTOR] Connection failed."); time.sleep(2.0); return
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1); self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0]); self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.connected = True; print("[COLLECTOR] Connected. Collecting frames.")
    def start(self):
        thread = threading.Thread(target=self._collector_thread, args=(), daemon=True); thread.start(); return self
    def get_frame(self):
        with self.lock:
            if self.frame_queue: return self.frame_queue.popleft()
            return None
    def get_latest_frame(self):
        with self.lock: return self.latest_frame
    def stop(self):
        self.stopped = True; time.sleep(0.5)
        if self.stream: self.stream.release()

from door_counter_engine import PolygonCounter # Assuming it's in a separate file

CONFIG = {}
STATE = {
    "camera_buffer": None, "is_processing": False, "counter": None,
    "processing_thread": None, "lock": threading.Lock(),
    "latest_processed_frame": None,
    "latest_counts": {'enter': 0, 'exit': 0, 'inside': 0, 'fps': 0.0}
}

# --- Background Processing Loop (Unchanged) ---
def processing_loop():
    fps_ema = 0.0
    while True:
        with STATE["lock"]:
            if not STATE["is_processing"]: break
        t_start = time.time()
        with STATE["lock"]:
            buffer = STATE["camera_buffer"]; counter = STATE["counter"]
        if buffer is None or counter is None:
            time.sleep(0.1); continue
        frame = buffer.get_frame()
        if frame is None:
            time.sleep(0.01); continue
            
        resolution = tuple(CONFIG.get("resolution", (1280, 720)))
        frame = cv2.resize(frame, resolution)
        processed_frame, counts = counter.process_frame(frame)
        
        t_end = time.time(); proc_time = t_end - t_start
        if proc_time > 0: fps_ema = (0.9 * fps_ema + 0.1 * (1.0 / proc_time))
        with STATE["lock"]:
            STATE["latest_processed_frame"] = processed_frame
            STATE["latest_counts"] = {'enter': counts['enter'], 'exit': counts['exit'], 'inside': counts['inside'], 'fps': round(fps_ema, 2)}
    print("Processing thread has finished.")

# --- FastAPI App Definition ---
app = FastAPI()

class StartRequest(BaseModel):
    video_source: str
    conf_threshold: float
    k_frames: int
    inside_definition: str

@app.post("/start")
def start_service(request: StartRequest):
    # ... (This endpoint is unchanged and correct)
    with STATE["lock"]:
        if STATE["is_processing"]:
            raise HTTPException(status_code=400, detail="Processing is already running.")
        regions_file = "regions.txt"
        if not os.path.exists(regions_file):
            raise HTTPException(status_code=400, detail="regions.txt not found. Please define regions first.")
        with open(regions_file, "r") as f:
            lines = f.readlines()
            # Using regex for safe parsing
            r1_line = lines[0].split(":")[1].strip(); r1_matches = re.findall(r'\((\d+),(\d+)\)', r1_line); r1_pts = [(int(x), int(y)) for x, y in r1_matches]
            r2_line = lines[1].split(":")[1].strip(); r2_matches = re.findall(r'\((\d+),(\d+)\)', r2_line); r2_pts = [(int(x), int(y)) for x, y in r2_matches]
        counter = PolygonCounter(model_path=CONFIG['model_path'], conf_threshold=request.conf_threshold, k_frames=request.k_frames)
        counter.set_regions(r1_pts, r2_pts, request.inside_definition)
        buffer = CameraBuffer(request.video_source, tuple(CONFIG.get("resolution")), buffer_size=128)
        buffer.start()
        STATE["camera_buffer"] = buffer; STATE["counter"] = counter; STATE["is_processing"] = True
        STATE["processing_thread"] = threading.Thread(target=processing_loop, daemon=True)
        STATE["processing_thread"].start()
    return {"message": "Processing started successfully."}

# --- MODIFIED: The /stop endpoint is now idempotent ---
@app.post("/stop")
def stop_service():
    with STATE["lock"]:
        # --- THIS IS THE FIX ---
        # If it's not running, just report that it's stopped. Don't raise an error.
        if not STATE["is_processing"]:
            return {"message": "Processing was already stopped."}
        
        STATE["is_processing"] = False # Signal the thread to stop
    
    # Wait for the thread to finish cleanly
    thread = STATE.get("processing_thread")
    if thread and thread.is_alive():
        thread.join()
    
    # Clean up resources
    with STATE["lock"]:
        if STATE["camera_buffer"]:
            STATE["camera_buffer"].stop()
        STATE.update({"camera_buffer": None, "counter": None})
        
    return {"message": "Processing stopped successfully."}

@app.get("/status")
def get_status():
    # ... (This endpoint is unchanged and correct)
    with STATE["lock"]:
        is_running = STATE["is_processing"]
        counts = STATE["latest_counts"]
        frame = STATE["latest_processed_frame"]
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            frame_base64 = None
    return {"is_running": is_running, "counts": counts, "frame": frame_base64}

# --- Main Execution ---
if __name__ == "__main__":
    try:
        with open("config.json", "r") as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it. Exiting.")
        exit()
    import re # Make sure re is imported for the start endpoint
    print("Starting background engine service...")
    uvicorn.run(app, host="0.0.0.0", port=8001)