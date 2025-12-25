import cv2
import time
import numpy as np
import threading
import json
import os
import re
import logging
from collections import deque
from datetime import datetime
import requests
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraBuffer:
    """Handles video stream connection and frame buffering"""
    def __init__(self, src, resolution, buffer_size=128):
        self.src = src
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.stopped = False
        self.stream = None
        self.frame_queue = deque(maxlen=buffer_size)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.connected = False

    def _collector_thread(self):
        while not self.stopped:
            if not self.connected:
                self.connect()
                continue
            grabbed, frame = self.stream.read()
            if not grabbed:
                logger.warning("[COLLECTOR] Lost connection. Reconnecting...")
                self.stream.release()
                self.connected = False
                time.sleep(2.0)
                continue
            with self.lock:
                self.frame_queue.append(frame)
                self.latest_frame = frame

    def connect(self):
        logger.info(f"[COLLECTOR] Connecting to: {self.src}...")
        self.stream = cv2.VideoCapture(self.src)
        if not self.stream.isOpened():
            logger.error("[COLLECTOR] Connection failed.")
            time.sleep(2.0)
            return
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.connected = True
        logger.info("[COLLECTOR] Connected. Collecting frames.")

    def start(self):
        thread = threading.Thread(target=self._collector_thread, args=(), daemon=True)
        thread.start()
        return self

    def get_frame(self):
        with self.lock:
            if self.frame_queue:
                return self.frame_queue.popleft()
            return None

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.stopped = True
        time.sleep(0.5)
        if self.stream:
            self.stream.release()


class PushAlertNotifier:
    """Handles HTTP Push Alerts to AutoProcAI"""
    def __init__(self, config: dict):
        self.url = config.get("url", "https://api-gw.autoprocai.com/smartaihub/push_alert")
        self.camera_id = config.get("camera_id", "")
        self.ai_module_id = config.get("ai_module_id", "")
        self.base_title = config.get("title", "Door Alert")
        self.zone_id = config.get("zone_id", "")
        
        self.last_notify_time = {}
        self.notify_cooldown = config.get("cooldown", 30)  # Seconds

    def send_alert(self, frame: np.ndarray, title_suffix: str = "") -> bool:
        """Encodes frame and sends POST request"""
        try:
            # 1. Encode image to JPG bytes
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.error("[PUSH] Failed to encode frame")
                return False
            img_bytes = buffer.tobytes()

            # 2. Prepare Data Payload
            # Appending suffix (e.g., " - ENTER") to title so the dashboard shows what happened
            full_title = f"{self.base_title} {title_suffix}".strip()
            
            data = {
                "camera_id": self.camera_id,
                "ai_module_id": self.ai_module_id,
                "title": full_title,
                "status": "auto",
                "zone_id": self.zone_id
            }

            # 3. Prepare Files
            # API requires a video_file. We send a dummy empty bytes object with mp4 extension
            # to satisfy the requirement without creating a file on disk.
            files = {
                "img_file": ("alert.jpg", img_bytes, "image/jpeg"),
                "video_file": ("dummy.mp4", b'', "video/mp4")
            }

            # 4. Send Request
            response = requests.post(self.url, data=data, files=files, timeout=10)
            
            if response.status_code == 200:
                resp_json = response.json()
                logger.info(f"[PUSH] Alert sent successfully: {resp_json}")
                return True
            else:
                logger.error(f"[PUSH] Failed to send alert. Code: {response.status_code}, Resp: {response.text}")
                return False

        except Exception as e:
            logger.error(f"[PUSH] Exception during send: {e}")
            return False

    def notify_event(self, event_type: str, frame: np.ndarray) -> bool:
        """Notify entry/exit events with cooldown"""
        now = time.time()
        last_time = self.last_notify_time.get(event_type, 0)

        # Skip if cooldown hasn't passed
        if not (now - last_time >= self.notify_cooldown):
            return False

        logger.info(f"[PUSH] Triggering alert for {event_type}")
        
        if event_type == 'ENTER':
            suffix = "- Person ENTERED"
        elif event_type == 'EXIT':
            suffix = "- Person EXITED"
        else:
            suffix = ""

        success = self.send_alert(frame, suffix)

        if success:
            self.last_notify_time[event_type] = now

        return success


class DoorCounterBackend:
    """Main backend service"""
    def __init__(self, config_file: str = "config.json", regions_file: str = "regions.txt"):
        self.config = self._load_config(config_file)
        self.regions = self._load_regions(regions_file)
        self.camera_buffer: Optional[CameraBuffer] = None
        self.counter = None
        self.notifier: Optional[PushAlertNotifier] = None
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.fps_counter = 0
        self.fps_time = time.time()
        self.fps = 0.0

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Config loaded: {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _load_regions(self, regions_file: str) -> dict:
        """Load predefined regions from text file"""
        if not os.path.exists(regions_file):
            logger.error(f"Regions file not found: {regions_file}")
            raise FileNotFoundError(f"Regions file not found: {regions_file}")

        try:
            regions = {}
            with open(regions_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if not line.strip() or ':' not in line:
                    continue
                region_name, coords_str = line.split(':', 1)
                region_name = region_name.strip()
                # Parse coordinates using regex: (x,y)
                matches = re.findall(r'\((\d+),(\d+)\)', coords_str)
                points = [(int(x), int(y)) for x, y in matches]
                if points:
                    regions[region_name] = points

            logger.info(f"Regions loaded: {list(regions.keys())}")
            return regions
        except Exception as e:
            logger.error(f"Error loading regions: {e}")
            raise

    def _initialize_notifier(self) -> bool:
        """Initialize Push Alert Notifier from config"""
        alert_config = self.config.get('push_alert', {})
        camera_id = alert_config.get('camera_id')
        ai_module_id = alert_config.get('ai_module_id')

        if not camera_id or not ai_module_id:
            logger.warning("Push Alert credentials (camera_id/ai_module_id) not configured. Notifications disabled.")
            return False

        try:
            self.notifier = PushAlertNotifier(alert_config)
            logger.info("Push Alert Notifier initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing Notifier: {e}")
            return False

    def start(self, video_source: Optional[str] = None) -> bool:
        """Start the backend service"""
        if self.is_running:
            logger.warning("Backend is already running")
            return False

        # Use video source from config or parameter
        source = video_source or self.config.get('auto_start_source')
        if not source:
            logger.error("No video source specified in config or parameters")
            return False

        # Import door counter engine
        try:
            from door_counter_engine import PolygonCounter
        except ImportError:
            logger.error("Cannot import PolygonCounter from door_counter_engine")
            return False

        try:
            with self.lock:
                # Initialize Notification System
                self._initialize_notifier()

                # Initialize counter
                model_path = self.config.get('model_path', 'yolov8n.pt')
                conf_threshold = self.config['default_settings'].get('conf_threshold', 0.35)
                k_frames = self.config['default_settings'].get('k_frames', 2)

                self.counter = PolygonCounter(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    k_frames=k_frames
                )

                # Set regions
                region1_pts = self.regions.get('region1', [])
                region2_pts = self.regions.get('region2', [])
                inside_definition = self.config['default_settings'].get('inside_definition', 'Region 1 is \'Inside\'')

                if not region1_pts or not region2_pts:
                    logger.error("Regions not properly loaded")
                    return False

                self.counter.set_regions(region1_pts, region2_pts, inside_definition)

                # Initialize camera buffer
                resolution = tuple(self.config.get('resolution', [1280, 720]))
                self.camera_buffer = CameraBuffer(source, resolution)
                self.camera_buffer.start()

                # Start processing
                self.is_running = True
                self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
                self.processing_thread.start()

            logger.info("Backend started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting backend: {e}")
            self.is_running = False
            return False

    def _processing_loop(self):
        """Main processing loop"""
        prev_enter_count = 0
        prev_exit_count = 0

        while self.is_running:
            try:
                if self.camera_buffer is None or self.counter is None:
                    time.sleep(0.1)
                    continue

                frame = self.camera_buffer.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Resize frame
                resolution = tuple(self.config.get('resolution', [1280, 720]))
                frame = cv2.resize(frame, resolution)

                # Process frame
                t_start = time.time()
                processed_frame, counts = self.counter.process_frame(frame)
                t_end = time.time()

                # Update FPS
                self.fps_counter += 1
                if t_end - self.fps_time >= 1.0:
                    self.fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_time = t_end
                    logger.info(f"FPS: {self.fps} | Enter: {counts['enter']} | Exit: {counts['exit']} | Inside: {counts['inside']}")

                # Check for new entry/exit events and push alerts
                if self.notifier:
                    if counts['enter'] > prev_enter_count:
                        # Use processed_frame (with bounding boxes) for the alert
                        self.notifier.notify_event('ENTER', processed_frame)
                        prev_enter_count = counts['enter']

                    if counts['exit'] > prev_exit_count:
                        self.notifier.notify_event('EXIT', processed_frame)
                        prev_exit_count = counts['exit']

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def stop(self):
        """Stop the backend service"""
        if not self.is_running:
            logger.warning("Backend is not running")
            return False

        try:
            with self.lock:
                self.is_running = False

            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)

            # Cleanup
            with self.lock:
                if self.camera_buffer:
                    self.camera_buffer.stop()
                self.camera_buffer = None
                self.counter = None

            logger.info("Backend stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping backend: {e}")
            return False

    def get_status(self) -> dict:
        """Get current status"""
        with self.lock:
            return {
                'is_running': self.is_running,
                'fps': self.fps,
                'camera_connected': self.camera_buffer.connected if self.camera_buffer else False,
                'push_alerts_enabled': self.notifier is not None
            }


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Door Counter Backend (Push Alert) - Starting...")
    logger.info("=" * 60)

    backend = DoorCounterBackend()

    try:
        # Start the backend
        if backend.start():
            logger.info("Backend is now running. Press Ctrl+C to stop.")
            # Keep the main thread alive
            while True:
                time.sleep(1)
                status = backend.get_status()
                if not status['is_running']:
                    break
        else:
            logger.error("Failed to start backend")
            return 1

    except KeyboardInterrupt:
        logger.info("Interrupt received. Stopping backend...")
        backend.stop()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

    logger.info("Backend shutdown complete")
    return 0


if __name__ == "__main__":
    exit(main())