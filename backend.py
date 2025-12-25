"""
Minimalist Backend Service for Door Counter
- Loads predefined regions and config from files
- Processes video stream and tracks door entry/exit
- Sends notifications to Telegram
- No UI/Gradio dependency
"""

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


class TelegramNotifier:
    """Handles Telegram notifications"""
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_notify_time = {}
        self.notify_cooldown = 30  # Minimum seconds between same type of notifications

    def send_message(self, message: str, skip_cooldown: bool = False) -> bool:
        """Send a text message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                logger.info(f"[TELEGRAM] Message sent: {message[:50]}...")
                return True
            else:
                logger.error(f"[TELEGRAM] Failed to send message: {response.text}")
                return False
        except Exception as e:
            logger.error(f"[TELEGRAM] Exception: {e}")
            return False

    def notify_event(self, event_type: str, track_id: int, current_count: int, frame: np.ndarray = None) -> bool:
        """Notify entry/exit events with cooldown to avoid spam"""
        now = time.time()
        last_time = self.last_notify_time.get(event_type, 0)

        # Skip if cooldown hasn't passed
        if not (now - last_time >= self.notify_cooldown):
            return False

        if event_type == 'ENTER':
            message = f"🚪 <b>Person ENTERED</b>\nID: {track_id}\n👥"
        elif event_type == 'EXIT':
            message = f"🚪 <b>Person EXITED</b>\nID: {track_id}\n👥"
        else:
            return False

        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send with image if available
        success = False
        if frame is not None:
            success = self.send_photo_with_message(frame, message)
        else:
            success = self.send_message(message)

        if success:
            self.last_notify_time[event_type] = now

        return success

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo to Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, 'rb') as f:
                files = {'photo': f}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                }
                response = requests.post(url, files=files, data=data, timeout=10)
                if response.status_code == 200:
                    logger.info(f"[TELEGRAM] Photo sent: {caption[:50]}...")
                    return True
                else:
                    logger.error(f"[TELEGRAM] Failed to send photo: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"[TELEGRAM] Exception: {e}")
            return False

    def send_photo_with_message(self, frame: np.ndarray, caption: str = "") -> bool:
        """Send a photo from numpy array to Telegram"""
        temp_path = "/tmp/door_counter_frame.jpg"
        try:
            # Save frame to temp file
            cv2.imwrite(temp_path, frame)
            # Send the photo
            success = self.send_photo(temp_path, caption)
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return success
        except Exception as e:
            logger.error(f"[TELEGRAM] Error sending photo: {e}")
            return False


class DoorCounterBackend:
    """Main backend service"""
    def __init__(self, config_file: str = "config.json", regions_file: str = "regions.txt"):
        self.config = self._load_config(config_file)
        self.regions = self._load_regions(regions_file)
        self.camera_buffer: Optional[CameraBuffer] = None
        self.counter = None
        self.telegram: Optional[TelegramNotifier] = None
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

    def _initialize_telegram(self) -> bool:
        """Initialize Telegram notifier from config"""
        telegram_config = self.config.get('telegram', {})
        bot_token = telegram_config.get('bot_token')
        chat_id = telegram_config.get('chat_id')

        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not configured. Notifications disabled.")
            return False

        try:
            self.telegram = TelegramNotifier(bot_token, chat_id)
            # Send startup message
            self.telegram.send_message("✅ Door Counter Backend Started")
            logger.info("Telegram notifier initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing Telegram: {e}")
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
                # Initialize Telegram
                self._initialize_telegram()

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
            # if self.telegram:
            #     self.telegram.send_message(f"🎥 Video Source: {source[:50]}...")
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

                # Check for new entry/exit events with frame capture
                if self.telegram:
                    send_image = self.config.get('telegram', {}).get('send_images', True)
                    
                    if counts['enter'] > prev_enter_count:
                        frame_to_send = processed_frame if send_image else None
                        self.telegram.notify_event('ENTER', 0, counts['inside'], frame_to_send)
                        prev_enter_count = counts['enter']

                    if counts['exit'] > prev_exit_count:
                        frame_to_send = processed_frame if send_image else None
                        self.telegram.notify_event('EXIT', 0, counts['inside'], frame_to_send)
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

            if self.telegram:
                self.telegram.send_message("🛑 Door Counter Backend Stopped")

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
                'telegram_enabled': self.telegram is not None
            }


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Door Counter Backend - Starting...")
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
