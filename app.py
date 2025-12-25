# app.py

import gradio as gr
import cv2
import time
import numpy as np
import os
import re
import requests
import base64

# --- Constants ---
ENGINE_API_URL = "http://127.0.0.1:8001" # The address of your engine service
REGIONS_FILENAME = "regions.txt"
UI_RESOLUTION = (1280, 720) # Must match the resolution in your config.json

# --- Global State for UI ONLY ---
STATE = {
    "current_image": None,
    "region_1_pts": [],
    "region_2_pts": [],
}

# --- UI Functions that call the Engine API ---
def start_processing(video_source, video_path, webcam_id_val, stream_url, conf_thresh, skip_frames, inside_definition):
    # This function now gathers all data and sends it to the engine's /start endpoint
    source_path = None
    if video_source == "Video File": source_path = video_path.name if video_path else None
    elif video_source == "Webcam": source_path = int(webcam_id_val)
    elif video_source == "Stream URL": source_path = stream_url
    if source_path is None: raise gr.Error("No valid video source provided.")

    payload = {
        "video_source": source_path,
        "conf_threshold": conf_thresh,
        "k_frames": skip_frames,
        "inside_definition": inside_definition
    }
    try:
        response = requests.post(f"{ENGINE_API_URL}/start", json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        return f"Engine Service: {response.json().get('message')}"
    except requests.exceptions.RequestException as e:
        return f"Error starting engine: {e}"

def stop_processing():
    try:
        response = requests.post(f"{ENGINE_API_URL}/stop")
        response.raise_for_status()
        return f"Engine Service: {response.json().get('message')}"
    except requests.exceptions.RequestException as e:
        return f"Error stopping engine: {e}"

def update_ui():
    # This is the main loop that gets data from the engine service
    while True:
        try:
            response = requests.get(f"{ENGINE_API_URL}/status", timeout=1)
            response.raise_for_status()
            data = response.json()

            frame_b64 = data.get("frame")
            frame = None
            if frame_b64:
                img_bytes = base64.b64decode(frame_b64)
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            yield {
                video_output: frame if frame is not None else None,
                enter_count_num: data["counts"]["enter"],
                exit_count_num: data["counts"]["exit"],
                inside_count_num: data["counts"]["inside"],
                fps_num: data["counts"]["fps"],
                task_status_text: "Processing..." if data["is_running"] else "Idle (Engine Running)"
            }
        except requests.exceptions.RequestException:
            # This happens if the engine service is not running
            yield {
                task_status_text: "Engine service is offline. Start engine_service.py.",
                video_output: None
            }
        
        time.sleep(1/20) # Refresh rate for the UI

# --- Local UI Functions for Setup ---
def load_source_for_preview(source, video_path, webcam_id, stream_url):
    # This function is unchanged and remains local to the UI
    reset_regions()
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
    STATE["current_image"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yield {drawing_canvas: STATE["current_image"], status_text: "Click 4 points for Region 1 or Load Regions.", load_source_button: gr.update(interactive=True)}

def handle_region_click(evt: gr.SelectData): # Unchanged
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

def save_regions_to_file(): # Unchanged
    if len(STATE["region_1_pts"]) != 4 or len(STATE["region_2_pts"]) != 4: return
    try:
        with open(REGIONS_FILENAME, "w") as f:
            r1_str = ",".join([f"({p[0]},{p[1]})" for p in STATE["region_1_pts"]]); f.write(f"region1: {r1_str}\n")
            r2_str = ",".join([f"({p[0]},{p[1]})" for p in STATE["region_2_pts"]]); f.write(f"region2: {r2_str}\n")
        print(f"Regions saved to {REGIONS_FILENAME}")
    except Exception as e: print(f"Error saving regions: {e}")

def load_regions_from_file(): # Unchanged
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
    except Exception as e: raise gr.Error(f"Failed to parse regions file: {e}")
    image_with_regions = STATE["current_image"].copy()
    cv2.polylines(image_with_regions, [np.array(STATE["region_1_pts"])], True, (0, 0, 255), 3)
    cv2.polylines(image_with_regions, [np.array(STATE["region_2_pts"])], True, (0, 255, 0), 3)
    status = "Regions loaded. Choose which is 'INSIDE'."; choices = ["Region 1 is 'Inside'", "Region 2 is 'Inside'"]
    return {drawing_canvas: image_with_regions, status_text: status, inside_definition_radio: gr.update(choices=choices, value=choices[1], interactive=True), start_button: gr.update(interactive=True)}

def reset_regions():
    STATE["region_1_pts"] = []; STATE["region_2_pts"] = []; STATE["current_image"] = None

def reset_all(): # Now it just resets the UI
    stop_processing() # Signal the service to stop
    reset_regions()
    return {
        video_output: None, drawing_canvas: None, status_text: "Awaiting video...",
        task_status_text: "Idle.", enter_count_num: 0, exit_count_num: 0,
        inside_count_num: 0, fps_num: 0.0, start_button: gr.update(interactive=True),
        inside_definition_radio: gr.update(choices=[], interactive=False, value=None),
        stream_url_input: gr.update(value=""),
    }

with gr.Blocks(theme=gr.themes.Soft(), title="Persistent Counter UI") as demo:
    gr.Markdown("# Persistent Smart Region Counter (UI)")
    # ... (The UI Layout is exactly the same as before)
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

    # --- Event Handling ---
    def update_source_ui(source): return {video_input_file: gr.update(visible=(source=="Video File")), webcam_input_id: gr.update(visible=(source=="Webcam")), stream_url_input: gr.update(visible=(source=="Stream URL"))}
    video_source_radio.change(fn=update_source_ui, inputs=video_source_radio, outputs=[video_input_file, webcam_input_id, stream_url_input])
    load_source_button.click(fn=load_source_for_preview, inputs=[video_source_radio, video_input_file, webcam_input_id, stream_url_input], outputs=[drawing_canvas, status_text, inside_definition_radio, load_source_button])
    drawing_canvas.select(fn=handle_region_click, inputs=[], outputs=[status_text, drawing_canvas, inside_definition_radio, start_button])
    load_regions_button.click(fn=load_regions_from_file, inputs=[], outputs=[drawing_canvas, status_text, inside_definition_radio, start_button])
    start_button.click(fn=start_processing, inputs=[video_source_radio, video_input_file, webcam_input_id, stream_url_input, conf_thresh_slider, skip_frames_slider, inside_definition_radio], outputs=[task_status_text])
    stop_button.click(fn=stop_processing, outputs=[task_status_text])
    reset_button.click(fn=reset_all, outputs=[video_output, drawing_canvas, status_text, task_status_text, enter_count_num, exit_count_num, inside_count_num, fps_num, start_button, inside_definition_radio, stream_url_input])
    
    # The 'See Running Process' button is now the default behavior on load
    demo.load(fn=update_ui, inputs=[], outputs=[video_output, enter_count_num, exit_count_num, inside_count_num, fps_num, task_status_text])

if __name__ == "__main__":
    demo.launch(debug=False, server_name="0.0.0.0", server_port=5170) # Set debug=False for production