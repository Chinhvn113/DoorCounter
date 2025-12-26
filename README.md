# AutoProcAI Door Counter Backend (Edge Service)

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.9%2B-green.svg) ![Status](https://img.shields.io/badge/status-production--ready-success.svg)

## 📖 Executive Summary

The **AutoProcAI Door Counter Backend** is a high-performance, edge-computing solution designed to perform real-time people counting and zone-based monitoring. Utilizing **YOLOv8** for object detection and a custom polygon-based tracking engine, this service processes video streams locally and transmits actionable event data (including visual snapshots) to the AutoProcAI Cloud via HTTP Push Alerts.

Designed for industrial reliability, the system features threaded video buffering, auto-reconnection logic, and bandwidth-efficient event notification.

---

## 🚀 Key Features

*   **Real-Time Edge Processing:** Low-latency detection using YOLOv8 models.
*   **Polygon Zone Logic:** precise entry/exit counting based on user-defined virtual tripwires (regions).
*   **Resilient Video Pipeline:**
    *   Threaded `CameraBuffer` ensures processing never blocks frame acquisition.
    *   Automatic reconnection logic for unstable RTSP/HTTP streams.
*   **Smart Notification System:**
    *   **Event-Driven:** Pushes alerts only when specific conditions (Enter/Exit) occur.
    *   **Visual Verification:** Attaches a high-res JPEG snapshot of the event.
    *   **Startup Health Check:** Sends a debug alert immediately upon service initialization.
    *   **Cooldown Management:** Prevents alert spamming via configurable timers.
*   **Lightweight Integration:** JSON-based configuration and standard HTTP POST API webhooks.

---

## 🛠️ System Architecture

1.  **Video Ingestion:** Connects to IP Cameras (RTSP) or USB devices.
2.  **Buffering:** Decouples frame reading from frame processing to maximize FPS.
3.  **Inference Engine:** Runs detection and tracks centroids across defined regions.
4.  **Logic Layer:** Determines `ENTER` vs `EXIT` events based on vector movement between regions.
5.  **Communications:** Dispatches multipart/form-data requests to the API Gateway.

---

## 📋 Prerequisites

### Hardware
*   **CPU:** Modern Multi-core CPU (Intel i5/i7 or ARM64/Jetson).
*   **RAM:** Minimum 4GB (8GB recommended).
*   **GPU (Optional):** NVIDIA GPU with CUDA support for high-FPS processing.

### Software
*   Python 3.9+
*   OpenCV (`headless` version recommended for server environments)
*   Ultralytics YOLO
*   AutoProcAI `door_counter_engine` module (Proprietary)

---

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/autoprocai/door-counter-edge.git
    cd door-counter-edge
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Sample `requirements.txt`:*
    ```text
    opencv-python-headless
    numpy
    requests
    ultralytics
    ```

3.  **Verify Model**
    Ensure `yolov8n.pt` (or your custom model) is present in the root directory.

---

## ⚙️ Configuration

The system relies on two primary configuration files: `config.json` and `regions.txt`.

### 1. `config.json` (System Settings)
Controls hardware connection, AI parameters, and API credentials.

```json
{
  "auto_start_source": "rtsp://user:pass@192.168.1.10:554/stream",
  "model_path": "yolov8n.pt",
  "resolution": [1280, 720],
  "default_settings": {
    "conf_threshold": 0.45,
    "k_frames": 2,
    "inside_definition": "Region 1 is 'Inside'"
  },
  "push_alert": {
    "url": "https://api-gw.autoprocai.com/smartaihub/push_alert",
    "camera_id": "CAM_WAREHOUSE_01",
    "ai_module_id": "MOD_PEOPLE_COUNT_V2",
    "zone_id": "ZONE_LOADING_DOCK",
    "title": "Warehouse Entry",
    "cooldown": 15
  }
}
```

### 2. `regions.txt` (Zone Definitions)
Defines the polygon points for tracking. Must contain at least two regions.
*Format: `RegionName: (x,y), (x,y), ...`*

```text
region1: (100,400), (500,400), (500,600), (100,600)
region2: (100,200), (500,200), (500,390), (100,390)
```

---

## 📡 API Integration (Webhooks)

When an event triggers, the backend sends a **POST** request with `Content-Type: multipart/form-data`.

### Request Payload

| Field | Type | Description |
| :--- | :--- | :--- |
| `camera_id` | String | Unique ID of the camera source. |
| `ai_module_id` | String | ID of the AI module generating the alert. |
| `zone_id` | String | Location/Zone identifier. |
| `title` | String | Dynamic title (e.g., "Door Alert - Person ENTERED"). |
| `status` | String | Fixed value: `auto`. |
| `img_file` | File | **Binary JPG data**. The snapshot of the event. |

---

## 🖥️ Usage

### Running Locally
To start the backend service:

```bash
python main.py
```

### Startup Behavior
1.  **Logs:** The console will display initialization steps and connection status.
2.  **Debug Alert:** Within 5 seconds of starting, the system will send a "System STARTUP" push notification to verify API connectivity.
3.  **Loop:** The system will print FPS statistics every second to stdout.

### Stopping
Press `Ctrl+C` to initiate a graceful shutdown. The system will release video resources and join threads before exiting.

---

## 🐳 Docker Deployment (Recommended)

For industrial deployment, wrap the application in a Docker container.

```dockerfile
# Simple Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY . .

CMD ["python", "main.py"]
```

**Run command:**
```bash
docker run -d --restart always \
  --name door-counter \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/regions.txt:/app/regions.txt \
  door-counter:latest
```

---

## 🔍 Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **Connection Failed** | RTSP URL incorrect or camera offline. | Verify stream via VLC. Check `config.json`. |
| **0 FPS / Lag** | High resolution or CPU bottleneck. | Lower resolution in `config.json`. Enable CUDA if available. |
| **No Alerts Sent** | API URL incorrect or Auth missing. | Check logs for `[PUSH] Failed`. Verify `camera_id` is set. |
| **False Positives** | Regions overlapping or bad lighting. | Adjust coordinates in `regions.txt`. Increase `conf_threshold`. |
| **Import Error** | Missing engine file. | Ensure `door_counter_engine.py` is in the `PYTHONPATH`. |

---

## 📝 Logging & Monitoring

The application logs to `stdout` (Console) by default.
*   **INFO:** Normal operation, FPS stats, Connection events.
*   **WARNING:** Network instability, Dropped frames.
*   **ERROR:** API failures, File missing, Critical crashes.

*Recommendation:* Redirect output to a log file or use a service like `systemd` or `fluentd` to aggregate logs in production.

---

## ⚖️ License & Disclaimer

**Copyright © 2025 AutoProcAI.**

*This software is provided "as is". While designed for industrial use, accuracy depends on environmental factors (lighting, camera angle). It is not intended for safety-critical applications.*
