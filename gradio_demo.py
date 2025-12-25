# app.py

import gradio as gr
import cv2
import time
import numpy as np
from door_counter_engine import PolygonCounter
import threading
import os
from collections import deque
import re

# ==============================================================================
# === CAMERA BUFFER CLASS (Unchanged) ===
# ==============================================================================
class CameraBuffer:
    # ... (The CameraBuffer class is unchanged and correct)
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

# ==============================================================================
# === APPLICATION LOGIC ===
# ==============================================================================

UI_RESOLUTION = (1280, 720)
STATE = {
    "camera_buffer": None, "is_processing": False, "counter": None,
    "region_1_pts": [], "region_2_pts": [], "current_image": None,
    "processing_thread": None, "lock": threading.Lock(),
    "latest_processed_frame": None,
    "latest_counts": {'enter': 0, 'exit': 0, 'inside': 0, 'fps': 0.0}
}
REGIONS_FILENAME = "regions.txt"
# --- NEW: Define the log filename as a constant ---
EVENT_LOG_FILENAME = "event_log.csv"

def processing_loop(): # ... (Unchanged and correct)
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
        frame = cv2.resize(frame, UI_RESOLUTION)
        processed_frame, counts = counter.process_frame(frame)
        t_end = time.time(); proc_time = t_end - t_start
        if proc_time > 0: fps_ema = (0.9 * fps_ema + 0.1 * (1.0 / proc_time))
        with STATE["lock"]:
            STATE["latest_processed_frame"] = processed_frame
            STATE["latest_counts"] = {'enter': counts['enter'], 'exit': counts['exit'], 'inside': counts['inside'], 'fps': round(fps_ema, 2)}
    print("Processing thread has finished.")

# --- MODIFIED: Pass the log filename to the engine ---
def start_processing(video_source, video_path, webcam_id_val, stream_url, conf_thresh, skip_frames, inside_definition):
    if len(STATE["region_1_pts"]) != 4 or len(STATE["region_2_pts"]) != 4: raise gr.Error("Error: Both regions must be defined.")
    with STATE["lock"]:
        if STATE["is_processing"]: return "Processing already running."
        print("UI: Starting processing...")
        counter = PolygonCounter(
            model_path='best_lowres_ncnn_model',
            conf_threshold=conf_thresh,
            k_frames=skip_frames,
            log_filename=EVENT_LOG_FILENAME # Pass the filename here
        )
        counter.set_regions(STATE["region_1_pts"], STATE["region_2_pts"], inside_definition)
        source_path = None
        if video_source == "Video File": source_path = video_path.name if video_path else None
        elif video_source == "Webcam": source_path = int(webcam_id_val)
        elif video_source == "Stream URL": source_path = stream_url
        if source_path is None: raise gr.Error("No valid video source.")
        buffer = CameraBuffer(source_path, (1920, 1080), buffer_size=128)
        buffer.start()
        STATE["camera_buffer"] = buffer; STATE["counter"] = counter; STATE["is_processing"] = True
        STATE["processing_thread"] = threading.Thread(target=processing_loop, daemon=True)
        STATE["processing_thread"].start()
    return "Processing started. You can close this window."

# --- MODIFIED: Delete the log file on reset ---
def reset_all():
    stop_processing()
    with STATE["lock"]:
        _reset_regions_internal()
        STATE["latest_processed_frame"] = None
        STATE["latest_counts"] = {'enter': 0, 'exit': 0, 'inside': 0, 'fps': 0.0}
        
        # --- Add logic to delete the log file ---
        try:
            if os.path.exists(EVENT_LOG_FILENAME):
                os.remove(EVENT_LOG_FILENAME)
                print(f"Event log '{EVENT_LOG_FILENAME}' has been reset.")
        except Exception as e:
            print(f"Error removing event log file: {e}")
            
    return {
        video_output: None, drawing_canvas: None, status_text: "Awaiting video...",
        task_status_text: "Idle.", enter_count_num: 0, exit_count_num: 0,
        inside_count_num: 0, fps_num: 0.0, start_button: gr.update(interactive=True),
        inside_definition_radio: gr.update(choices=[], interactive=False, value=None),
        stream_url_input: gr.update(value=""),
    }

# --- All other functions are unchanged and correct ---
def stop_processing():
    with STATE["lock"]:
        if not STATE["is_processing"]: return "Not running."
        print("UI: Sending stop signal..."); STATE["is_processing"] = False
    thread = STATE.get("processing_thread")
    if thread and thread.is_alive():
        thread.join()
    with STATE["lock"]:
        if STATE["camera_buffer"]: STATE["camera_buffer"].stop(); STATE["camera_buffer"] = None
        STATE["counter"] = None
    return "Processing stopped."
    
def load_source_for_preview(source, video_path, webcam_id, stream_url):
    if STATE["is_processing"]: raise gr.Error("Cannot change source while processing is running.")
    _reset_regions_internal_wrapper()
    yield {drawing_canvas: None, status_text: f"Connecting...", load_source_button: gr.update(value="Loading...", interactive=False)}
    source_path = None
    if source == "Video File": source_path = video_path.name if video_path else None
    elif source == "Webcam": source_path = int(webcam_id)
    elif source == "Stream URL": source_path = stream_url
    if source_path is None: yield {status_text: "Please provide a valid source.", load_source_button: gr.update(interactive=True)}; return
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened(): yield {status_text: f"Error opening source.", load_source_button: gr.update(interactive=True)}; return
    ret, frame = cap.read()
    cap.release()
    if not ret: yield {status_text: "Could not read preview frame.", load_source_button: gr.update(interactive=True)}; return
    frame = cv2.resize(frame, UI_RESOLUTION)
    with STATE["lock"]: STATE["current_image"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yield {drawing_canvas: STATE["current_image"], status_text: "Click 4 points for Region 1 or Load Regions.", load_source_button: gr.update(interactive=True)}

def save_regions_to_file():
    with STATE["lock"]:
        if len(STATE["region_1_pts"]) != 4 or len(STATE["region_2_pts"]) != 4: return
        try:
            with open(REGIONS_FILENAME, "w") as f:
                r1_str = ",".join([f"({p[0]},{p[1]})" for p in STATE["region_1_pts"]]); f.write(f"region1: {r1_str}\n")
                r2_str = ",".join([f"({p[0]},{p[1]})" for p in STATE["region_2_pts"]]); f.write(f"region2: {r2_str}\n")
            print(f"Regions saved to {REGIONS_FILENAME}")
        except Exception as e: print(f"Error saving regions: {e}")

def load_regions_from_file():
    with STATE["lock"]:
        if STATE["current_image"] is None: raise gr.Error("Please load a video source first.")
        if not os.path.exists(REGIONS_FILENAME): raise gr.Error(f"Regions file not found: {REGIONS_FILENAME}")
        try:
            with open(REGIONS_FILENAME, "r") as f:
                lines = f.readlines()
                r1_line = lines[0].split(":")[1].strip(); r1_matches = re.findall(r'\((\d+),(\d+)\)', r1_line); r1_pts = [(int(x), int(y)) for x, y in r1_matches]
                r2_line = lines[1].split(":")[1].strip(); r2_matches = re.findall(r'\((\d+),(\d+)\)', r2_line); r2_pts = [(int(x), int(y)) for x, y in r2_matches]
                if len(r1_pts) != 4 or len(r2_pts) != 4: raise ValueError("File does not contain 4 points for each region.")
                STATE["region_1_pts"] = r1_pts; STATE["region_2_pts"] = r2_pts
            print(f"Regions loaded from {REGIONS_FILENAME}")
        except Exception as e: raise gr.Error(f"Failed to parse regions file. Please redefine them. Error: {e}")
    image_with_regions = STATE["current_image"].copy()
    cv2.polylines(image_with_regions, [np.array(STATE["region_1_pts"])], True, (0, 0, 255), 3)
    cv2.polylines(image_with_regions, [np.array(STATE["region_2_pts"])], True, (0, 255, 0), 3)
    status = "Regions loaded. Choose which is 'INSIDE'."; choices = ["Region 1 is 'Inside'", "Region 2 is 'Inside'"]
    return {drawing_canvas: image_with_regions, status_text: status, inside_definition_radio: gr.update(choices=choices, value=choices[1], interactive=True), start_button: gr.update(interactive=True)}

def handle_region_click(evt: gr.SelectData):
    if STATE["current_image"] is None: return "Load a video first.", None, gr.update(interactive=False), gr.update(interactive=False)
    clicked_point = (evt.index[0], evt.index[1])
    defining_region_1 = len(STATE["region_1_pts"]) < 4
    defining_region_2 = not defining_region_1 and len(STATE["region_2_pts"]) < 4
    if defining_region_1:
        STATE["region_1_pts"].append(clicked_point); points_drawn = len(STATE["region_1_pts"])
        image_with_drawing = STATE["current_image"].copy()
        for pt in STATE["region_1_pts"]: cv2.circle(image_with_drawing, pt, 7, (255, 0, 255), -1)
        if points_drawn < 4: return f"Defining Region 1: Point {points_drawn+1}/4...", image_with_drawing, gr.update(interactive=False), gr.update(interactive=False)
        else:
            cv2.polylines(image_with_drawing, [np.array(STATE["region_1_pts"])], True, (0, 0, 255), 3)
            return "Region 1 defined. Click 4 points for Region 2.", image_with_drawing, gr.update(interactive=False), gr.update(interactive=False)
    elif defining_region_2:
        STATE["region_2_pts"].append(clicked_point); points_drawn = len(STATE["region_2_pts"])
        image_with_drawing = STATE["current_image"].copy()
        cv2.polylines(image_with_drawing, [np.array(STATE["region_1_pts"])], True, (0, 0, 255), 3)
        for pt in STATE["region_2_pts"]: cv2.circle(image_with_drawing, pt, 7, (0, 255, 255), -1)
        if points_drawn < 4: return f"Defining Region 2: Point {points_drawn+1}/4...", image_with_drawing, gr.update(interactive=False), gr.update(interactive=False)
        else:
            cv2.polylines(image_with_drawing, [np.array(STATE["region_2_pts"])], True, (0, 255, 0), 3)
            save_regions_to_file()
            choices = ["Region 1 is 'Inside'", "Region 2 is 'Inside'"]
            return "All regions defined and saved. Choose 'INSIDE' region.", image_with_drawing, gr.update(choices=choices, value=choices[1], interactive=True), gr.update(interactive=True)
    return "All regions defined. Press Reset.", None, gr.update(), gr.update()

def update_ui():
    while True:
        with STATE["lock"]:
            frame = STATE["latest_processed_frame"]; counts = STATE["latest_counts"]
        if frame is not None:
            yield {video_output: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), enter_count_num: counts['enter'], exit_count_num: counts['exit'], inside_count_num: counts['inside'], fps_num: counts['fps']}
        time.sleep(1/30)

def _reset_regions_internal():
    STATE["region_1_pts"] = []; STATE["region_2_pts"] = []; STATE["current_image"] = None

def _reset_regions_internal_wrapper():
    with STATE["lock"]: _reset_regions_internal()

with gr.Blocks(theme=gr.themes.Soft(), title="Persistent Region Counter") as demo:
    # --- The UI Layout is unchanged ---
    gr.Markdown("# Persistent Smart Region Counter")
    with gr.Row():
        with gr.Column(scale=6):
            video_output = gr.Image(label="Live Video Feed", interactive=False, height=720)
        with gr.Column(scale=2):
            gr.Markdown("### Task Board")
            with gr.Tabs():
                with gr.TabItem("Setup"):
                    with gr.Accordion("1. Select Video Source", open=True):
                        video_source_radio = gr.Radio(["Video File", "Webcam", "Stream URL"], label="Source", value="Video File")
                        video_input_file = gr.File(label="Upload Video", file_types=['video'], visible=True)
                        webcam_input_id = gr.Dropdown([0, 1, 2, 3], label="Webcam ID", value=0, visible=False)
                        stream_url_input = gr.Textbox(label="Stream URL", placeholder="e.g., rtsp://...", visible=False)
                        load_source_button = gr.Button("Load Source for Preview")
                    with gr.Accordion("2. Define Regions & Direction", open=False):
                        status_text = gr.Markdown("Awaiting video source...")
                        drawing_canvas = gr.Image(label="Click 4 points per region", interactive=True, type="numpy")
                        load_regions_button = gr.Button("Load Regions from File")
                        inside_definition_radio = gr.Radio(choices=[], label="Define 'Inside' Area:", interactive=False)
                with gr.TabItem("Controls & Stats"):
                    start_button = gr.Button("Start Processing", variant="primary")
                    stop_button = gr.Button("Stop Processing", variant="stop")
                    reset_button = gr.Button("Reset All", variant="secondary")
                    task_status_text = gr.Textbox("Idle.", label="Status", interactive=False)
                    with gr.Row():
                        enter_count_num = gr.Number(label="Total Entered", value=0, interactive=False)
                        exit_count_num = gr.Number(label="Total Exited", value=0, interactive=False)
                    with gr.Row():
                        inside_count_num = gr.Number(label="Currently Inside", value=0, interactive=False)
                        fps_num = gr.Number(label="Processing FPS", value=0.0, interactive=False)
                    with gr.Row():
                        conf_thresh_slider = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence")
                        skip_frames_slider = gr.Slider(1, 10, value=1, step=1, label="Process 1/N Frames")
    
    def update_source_ui(source): return {video_input_file: gr.update(visible=(source=="Video File")), webcam_input_id: gr.update(visible=(source=="Webcam")), stream_url_input: gr.update(visible=(source=="Stream URL"))}
    video_source_radio.change(fn=update_source_ui, inputs=video_source_radio, outputs=[video_input_file, webcam_input_id, stream_url_input])
    load_source_button.click(fn=load_source_for_preview, inputs=[video_source_radio, video_input_file, webcam_input_id, stream_url_input], outputs=[drawing_canvas, status_text, inside_definition_radio, load_source_button])
    drawing_canvas.select(fn=handle_region_click, inputs=[], outputs=[status_text, drawing_canvas, inside_definition_radio, start_button])
    load_regions_button.click(fn=load_regions_from_file, inputs=[], outputs=[drawing_canvas, status_text, inside_definition_radio, start_button])
    start_button.click(fn=start_processing, inputs=[video_source_radio, video_input_file, webcam_input_id, stream_url_input, conf_thresh_slider, skip_frames_slider, inside_definition_radio], outputs=[task_status_text])
    stop_button.click(fn=stop_processing, outputs=[task_status_text])
    reset_button.click(fn=reset_all, outputs=[video_output, drawing_canvas, status_text, task_status_text, enter_count_num, exit_count_num, inside_count_num, fps_num, start_button, inside_definition_radio, stream_url_input])
    demo.load(fn=update_ui, inputs=[], outputs=[video_output, enter_count_num, exit_count_num, inside_count_num, fps_num])

if __name__ == "__main__":
    demo.launch(debug=True, server_name="0.0.0.0", server_port=5170)