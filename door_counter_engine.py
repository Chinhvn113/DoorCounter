# door_counter_engine.py

from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import cv2
import time
from scipy.optimize import linear_sum_assignment
import csv
from datetime import datetime
import os
# --- Helper function for tracker matching (no changes) ---
def iou_batch(bboxes1, bboxes2):
    bboxes1 = np.asarray(bboxes1)
    bboxes2 = np.asarray(bboxes2)
    if bboxes1.ndim == 1: bboxes1 = bboxes1[np.newaxis, :]
    if bboxes2.ndim == 1: bboxes2 = bboxes2[np.newaxis, :]
    x1_1, y1_1, x2_1, y2_1 = np.split(bboxes1, 4, axis=1)
    x1_2, y1_2, x2_2, y2_2 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x1_1, x1_2.T); yA = np.maximum(y1_1, y1_2.T)
    xB = np.minimum(x2_1, x2_2.T); yB = np.minimum(y2_1, y2_2.T)
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    boxA_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    boxB_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = interArea / (boxA_area + boxB_area.T - interArea)
    return iou

class DoorCounter:
    def __init__(self, model_path='yolov8n.pt', img_size=480, conf_threshold=0.35, k_frames=10):
        self.model = YOLO(model_path, task='detect')
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.k_frames = k_frames
        self.frame_count = 0

        # --- REVISED: Line Crossing Logic ---
        self.counting_line = None       # Stores (p1, p2)
        self.person_positions = {}      # {track_id: 'side_A' | 'side_B'}
        self.inside_side = None         # Will be set to 'side_A' or 'side_B' by the user

        self.enter_count = 0
        self.exit_count = 0
        
        # KCF Tracker Attributes
        self.trackers = {}
        self.next_track_id = 0
        self.lost_track_counts = defaultdict(int)

    def set_line(self, p1, p2, inside_side_label):
        """
        Sets the counting line and defines which side is 'inside'.
        - p1: Start point of the line (x1, y1)
        - p2: End point of the line (x2, y2)
        - inside_side_label: 'side_A' or 'side_B'
        """
        self.counting_line = (p1, p2)
        self.inside_side = inside_side_label
            
        print(f"Counting line set from {p1} to {p2}.")
        print(f"'{self.inside_side}' has been designated as the 'INSIDE' area.")
        self.reset_counters()

    def reset_counters(self):
        self.person_positions.clear()
        self.enter_count = 0
        self.exit_count = 0
        self.trackers.clear()
        self.lost_track_counts.clear()
        self.next_track_id = 0
        print("Counters and states have been reset.")

    def _get_position(self, bbox_center):
        """
        Determines which side of the line a point is on using the cross-product.
        The side labels ('side_A', 'side_B') are relative to the vector p1 -> p2.
        'side_A' is on the "left" of the vector, 'side_B' is on the "right".
        """
        if not self.counting_line:
            return "unknown"
            
        p1, p2 = self.counting_line
        cx, cy = bbox_center
        
        cross_product = (p2[0] - p1[0]) * (cy - p1[1]) - (p2[1] - p1[1]) * (cx - p1[0])
        
        if cross_product > 5:
            return 'side_A'
        elif cross_product < -5:
            return 'side_B'
        else:
            return "on_line"

    # --- KCF Tracker Logic (No changes needed) ---
    def _update_trackers(self, frame):
        updated_boxes = {}
        lost_track_ids = []
        for track_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                updated_boxes[track_id] = [x, y, x + w, y + h]
                self.lost_track_counts[track_id] = 0
            else:
                self.lost_track_counts[track_id] += 1
                if self.lost_track_counts[track_id] > 5:
                    lost_track_ids.append(track_id)
        for track_id in lost_track_ids:
            del self.trackers[track_id]
            if track_id in self.person_positions: del self.person_positions[track_id]
        return updated_boxes

    def _detect_and_match(self, frame, tracked_boxes):
        results = self.model(frame, classes=[0], verbose=False, imgsz=self.img_size, conf=self.conf_threshold)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            detected_boxes = results[0].boxes.xyxy.cpu().numpy()
            tracked_ids = list(tracked_boxes.keys())
            if len(tracked_boxes) > 0:
                iou_matrix = iou_batch(detected_boxes, list(tracked_boxes.values()))
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = {c for c in col_ind}
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] > 0.3:
                        track_id = tracked_ids[c]
                        bbox = detected_boxes[r]
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                        self.trackers[track_id] = tracker
                unmatched_detections = set(range(len(detected_boxes))) - set(row_ind)
                for i in unmatched_detections:
                    bbox = detected_boxes[i]
                    tracker = cv2.TrackerKCF_create(); tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    self.trackers[self.next_track_id] = tracker; self.next_track_id += 1
            else:
                for bbox in detected_boxes:
                    tracker = cv2.TrackerKCF_create(); tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    self.trackers[self.next_track_id] = tracker; self.next_track_id += 1


    def process_frame(self, frame):
        if self.counting_line:
            p1, p2 = self.counting_line
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

        self.frame_count += 1
        current_boxes = self._update_trackers(frame)
        if self.frame_count % self.k_frames == 0:
            self._detect_and_match(frame, current_boxes)
            current_boxes = {tid: [int(v) for v in tr.get_position()] for tid, tr in self.trackers.items() if hasattr(tr, 'get_position')}

        for track_id, box in current_boxes.items():
            x1, y1, w, h = box
            center_x, center_y = x1 + w / 2, y1 + h / 2
            
            last_position = self.person_positions.get(track_id, 'unknown')
            current_position = self._get_position((center_x, center_y))

            if current_position == "on_line": current_position = last_position
            
            # --- REVISED: State Transition Logic ---
            if last_position != 'unknown' and current_position != last_position:
                # An ENTRY is defined as moving FROM the outside TO the inside.
                if last_position != self.inside_side and current_position == self.inside_side:
                    self.enter_count += 1
                    print(f"✓ ENTER - ID:{track_id} | Total Inside: {self.enter_count - self.exit_count}")
                # An EXIT is defined as moving FROM the inside TO the outside.
                elif last_position == self.inside_side and current_position != self.inside_side:
                    self.exit_count += 1
                    print(f"✓ EXIT - ID:{track_id} | Total Inside: {self.enter_count - self.exit_count}")
            
            if current_position != 'unknown': self.person_positions[track_id] = current_position

            # --- REVISED: Visualization ---
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x1+w), int(y1+h)
            is_inside = (current_position == self.inside_side)
            color = (0, 255, 0) if is_inside else (0, 0, 255)
            label = f"ID:{track_id} {'IN' if is_inside else 'OUT'}"
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, label, (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (int(center_x), int(center_y)), radius=5, color=color, thickness=-1)
        
        counts = { 'enter': self.enter_count, 'exit': self.exit_count, 'inside': self.enter_count - self.exit_count }
        return frame, counts
    

class DoorCounterYoloTrack:
    """
    A version of the DoorCounter that uses the built-in YOLOv8 tracker (`model.track()`)
    and correctly implements frame skipping logic internally to reduce computational load.
    """
    def __init__(self, model_path='yolov8n.pt', img_size=480, conf_threshold=0.35, k_frames=1):
        """
        Initializes the counter.
        - k_frames: The interval for processing frames. e.g., if 3, it processes 1 of every 3 frames.
        """
        self.model = YOLO(model_path, task='detect')
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        
        # --- Frame Skipping Logic ---
        self.k_frames = k_frames
        self.frame_count = 0
        
        # --- State for Skipped Frames ---
        # Store the last known results to display on skipped frames for a smoother look
        self.last_known_boxes = []
        self.last_known_track_ids = []

        # --- Line Crossing Logic ---
        self.counting_line = None
        self.person_positions = {}
        self.inside_side = None
        self.enter_count = 0
        self.exit_count = 0

    def set_line(self, p1, p2, inside_side_label):
        self.counting_line = (p1, p2)
        self.inside_side = inside_side_label
        print(f"Line set from {p1} to {p2}. '{self.inside_side}' is 'INSIDE'.")
        self.reset_counters()

    def reset_counters(self):
        self.person_positions.clear()
        self.enter_count = 0
        self.exit_count = 0
        self.frame_count = 0
        self.last_known_boxes = []
        self.last_known_track_ids = []
        print("Counters and tracking states have been reset.")

    def _get_position(self, bbox_center):
        if not self.counting_line: return "unknown"
        p1, p2 = self.counting_line
        cx, cy = bbox_center
        cross_product = (p2[0] - p1[0]) * (cy - p1[1]) - (p2[1] - p1[1]) * (cx - p1[0])
        if cross_product > 5: return 'side_A'
        elif cross_product < -5: return 'side_B'
        else: return "on_line"

    def process_frame(self, frame):
        self.frame_count += 1
        
        # --- Core Frame Skipping Logic ---
        # Only run the expensive model.track() call on every k-th frame.
        if self.frame_count % self.k_frames == 0:
            results = self.model.track(
                frame, persist=True, classes=[0], verbose=False,
                imgsz=self.img_size, conf=self.conf_threshold, tracker="bytetrack.yaml"
            )

            # If results are found, update the last known state
            if results[0].boxes is not None and results[0].boxes.id is not None:
                self.last_known_boxes = results[0].boxes.xyxy.cpu().numpy()
                self.last_known_track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                # If no objects are tracked, clear the last known state
                self.last_known_boxes = []
                self.last_known_track_ids = []
        
        # The logic below runs on EVERY frame, using the last known data for skipped frames.
        
        # Process the latest tracking data (either new or from a previous frame)
        if len(self.last_known_track_ids) > 0:
            for box, track_id in zip(self.last_known_boxes, self.last_known_track_ids):
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                last_position = self.person_positions.get(track_id, 'unknown')
                current_position = self._get_position((center_x, center_y))

                if current_position == "on_line": current_position = last_position
                
                # Only update counts on the frame where the model actually ran to avoid multi-counting
                if self.frame_count % self.k_frames == 0:
                    if last_position != 'unknown' and current_position != last_position:
                        if last_position != self.inside_side and current_position == self.inside_side:
                            self.enter_count += 1
                        elif last_position == self.inside_side and current_position != self.inside_side:
                            self.exit_count += 1
                
                if current_position != 'unknown':
                    self.person_positions[track_id] = current_position

        # --- Visualization (runs on every frame) ---
        if self.counting_line:
            p1, p2 = self.counting_line
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

        # Draw the last known bounding boxes on the current frame
        if len(self.last_known_track_ids) > 0:
            for box, track_id in zip(self.last_known_boxes, self.last_known_track_ids):
                current_pos = self.person_positions.get(track_id, 'unknown')
                is_inside = (current_pos == self.inside_side)
                color = (0, 255, 0) if is_inside else (0, 0, 255)
                label = f"ID:{track_id} {'IN' if is_inside else 'OUT'}"
                ix1, iy1, ix2, iy2 = map(int, box)
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
                cv2.putText(frame, label, (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        counts = {
            'enter': self.enter_count,
            'exit': self.exit_count,
            'inside': self.enter_count - self.exit_count
        }
        
        return frame, counts


# door_counter_engine.py

class PolygonCounter:
    """
    A counter that intelligently adapts its tracking point based on the spatial layout
    of the 'inside' and 'outside' regions.
    """
    def __init__(self, model_path='yolov8n.pt', img_size=480, conf_threshold=0.35, k_frames=1, log_filename="event_log.csv"):
        self.model = YOLO(model_path, task='detect')
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.k_frames = k_frames
        self.frame_count = 0
        
        self.log_filename = log_filename
        self._initialize_log_file()

        self.last_known_boxes = []
        self.last_known_track_ids = []
        self.region_1 = None
        self.region_2 = None
        self.inside_region_name = None
        self.outside_region_name = None
        self.person_locations = {}
        self.enter_count = 0
        self.exit_count = 0

    def _initialize_log_file(self):
        if not os.path.exists(self.log_filename):
            try:
                with open(self.log_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Event', 'TrackID'])
                print(f"Log file created at '{self.log_filename}'")
            except Exception as e:
                print(f"Error initializing log file: {e}")
    
    def _log_event(self, event_type, track_id):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, event_type, track_id])
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def set_regions(self, region_1_pts, region_2_pts, inside_region_choice):
        self.region_1 = np.array(region_1_pts, dtype=np.int32)
        self.region_2 = np.array(region_2_pts, dtype=np.int32)
        if "1" in inside_region_choice:
            self.inside_region_name, self.outside_region_name = 'region_1', 'region_2'
        else:
            self.inside_region_name, self.outside_region_name = 'region_2', 'region_1'
        print(f"Regions set. '{self.inside_region_name}' is INSIDE.")
        self.reset_counters()

    def reset_counters(self):
        self.person_locations.clear()
        self.enter_count = 0; self.exit_count = 0; self.frame_count = 0
        self.last_known_boxes = []; self.last_known_track_ids = []
        print("Counters and states have been reset.")

    def _get_location(self, point):
        if self.region_1 is not None and cv2.pointPolygonTest(self.region_1, point, False) >= 0: return 'region_1'
        if self.region_2 is not None and cv2.pointPolygonTest(self.region_2, point, False) >= 0: return 'region_2'
        return 'none'

    def process_frame(self, frame):
        self.frame_count += 1

        if self.frame_count % self.k_frames == 0:
            results = self.model.track(
                frame, persist=True, classes=[0], verbose=False,
                imgsz=self.img_size, conf=self.conf_threshold, tracker="bytetrack.yaml"
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                current_boxes = results[0].boxes.xyxy.cpu().numpy()
                current_track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(current_boxes, current_track_ids):
                    x1, y1, x2, y2 = box
                    
                    # Always use the center of the bottom edge for tracking
                    tracking_point = (int((x1 + x2) / 2), int(y2))

                    current_location = self._get_location(tracking_point)
                    last_valid_region = self.person_locations.get(track_id, 'none')

                    if current_location != 'none' and current_location != last_valid_region:
                        if last_valid_region == self.outside_region_name and current_location == self.inside_region_name:
                            self.enter_count += 1
                            self._log_event('ENTER', track_id)
                        elif last_valid_region == self.inside_region_name and current_location == self.outside_region_name:
                            self.exit_count += 1
                            self._log_event('EXIT', track_id)
                        
                        self.person_locations[track_id] = current_location

                self.last_known_boxes = current_boxes; self.last_known_track_ids = current_track_ids
            else:
                self.last_known_boxes, self.last_known_track_ids = [], []
        
        # --- Visualization (unchanged) ---
        if self.region_1 is not None:
            color = (0, 255, 0) if self.inside_region_name == 'region_1' else (0, 0, 255)
            cv2.polylines(frame, [self.region_1], True, color, 3)
            cv2.putText(frame, "IN" if self.inside_region_name == 'region_1' else "OUT", (self.region_1[0][0], self.region_1[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if self.region_2 is not None:
            color = (0, 255, 0) if self.inside_region_name == 'region_2' else (0, 0, 255)
            cv2.polylines(frame, [self.region_2], True, color, 3)
            cv2.putText(frame, "IN" if self.inside_region_name == 'region_2' else "OUT", (self.region_2[0][0], self.region_2[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if len(self.last_known_track_ids) > 0:
            for box, track_id in zip(self.last_known_boxes, self.last_known_track_ids):
                location = self.person_locations.get(track_id, 'none')
                color = (255, 255, 255)
                if location == self.inside_region_name: color = (0, 255, 0)
                elif location == self.outside_region_name: color = (0, 0, 255)
                ix1, iy1, ix2, iy2 = map(int, box)
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (ix1, iy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Visualize the tracking point (center of bottom edge)
                t_point = (int((ix1+ix2)/2), iy2)
                cv2.circle(frame, t_point, 7, (255, 255, 0), -1) # Make the point bigger and more visible

        counts = {'enter': self.enter_count, 'exit': self.exit_count, 'inside': self.enter_count - self.exit_count}
        return frame, counts