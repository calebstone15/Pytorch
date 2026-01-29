import io
import os
import time
import logging
import atexit
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, H264Encoder
from picamera2.outputs import FileOutput
from threading import Condition, Lock
import json
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
logging.info(f"Recordings will be saved to: {RECORDINGS_DIR}")

CAMERA_PROFILES = {
    0: [
        {"id": "ai_720p25", "label": "1280 × 720 @ 25 fps", "main": (1280, 720), "lores": (800, 600), "fps": 25},
        {"id": "ai_1080p30", "label": "1920 × 1080 @ 30 fps", "main": (1920, 1080), "lores": (960, 540), "fps": 30},
        {"id": "ai_1296p40", "label": "2304 × 1296 @ 40 fps", "main": (2304, 1296), "lores": (1024, 768), "fps": 40},
    ],
    1: [
        {"id": "cm3_720p25", "label": "1280 × 720 @ 25 fps", "main": (1280, 720), "lores": (800, 600), "fps": 25},
        {"id": "cm3_1080p30", "label": "1920 × 1080 @ 30 fps", "main": (1920, 1080), "lores": (960, 540), "fps": 30},
        {"id": "cm3_1520p50", "label": "2028 × 1520 @ 50 fps", "main": (2028, 1520), "lores": (1014, 760), "fps": 50},
        {"id": "cm3_4608p15", "label": "4608 × 2592 @ 15 fps", "main": (4608, 2592), "lores": (1152, 648), "fps": 15},
    ],
}
camera_settings = {0: None, 1: None}
METRICS_WINDOW_SECONDS = 10
metrics_lock = Lock()
metrics_state = {
    cam: {"samples": deque(), "bytes_per_sec": 0.0, "fps": 0.0, "last_timestamp": None}
    for cam in camera_settings
}

def reset_metrics(cam_num):
    with metrics_lock:
        state = metrics_state.get(cam_num)
        if not state:
            return
        state["samples"].clear()
        state["bytes_per_sec"] = 0.0
        state["fps"] = 0.0
        state["last_timestamp"] = None

def record_metric(cam_num, payload_bytes):
    now = time.time()
    with metrics_lock:
        state = metrics_state.get(cam_num)
        if not state:
            return
        state["samples"].append((now, payload_bytes))
        while state["samples"] and now - state["samples"][0][0] > METRICS_WINDOW_SECONDS:
            state["samples"].popleft()
        window = max(now - state["samples"][0][0], 1e-6)
        total_bytes = sum(size for _, size in state["samples"])
        frames = len(state["samples"])
        state["bytes_per_sec"] = total_bytes / window
        state["fps"] = frames / window
        state["last_timestamp"] = now

# A thread-safe, buffering, file-like object for the streaming output
class StreamingOutput(io.BufferedIOBase):
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        record_metric(self.cam_id, len(buf))

# --- Flask App Initialization ---
app = Flask(__name__)

# --- PiCamera2 Initialization ---
picam0 = None
picam1 = None
stream_output0 = None
stream_output1 = None
video_encoder0 = None
video_encoder1 = None

# --- State Management ---
# We use locks to prevent race conditions when starting/stopping recordings
cam0_lock = Lock()
cam1_lock = Lock()
# State dictionary to hold recording status and filenames
cam0_state = {"recording": False, "filename": None}
cam1_state = {"recording": False, "filename": None}

def initialize_cameras():
    """Initializes all available cameras and starts their live streams."""
    global picam0, stream_output0, video_encoder0
    global picam1, stream_output1, video_encoder1
    
    camera_ids = [info['Id'] for info in Picamera2.global_camera_info()]
    logging.info(f"Found {len(camera_ids)} cameras: {camera_ids}")

    # --- Initialize Camera 0 (CVHAD) ---
    if len(camera_ids) >= 1:
        try:
            logging.info("Initializing camera 0 (CVHAD)...")
            default_profile0 = next((p for p in CAMERA_PROFILES[0] if p["id"] == "ai_720p25"), CAMERA_PROFILES[0][0])
            picam0 = Picamera2(camera_num=0)
            config = picam0.create_video_configuration(
                main={"size": default_profile0["main"]},
                lores={"size": default_profile0["lores"]},
                controls={"FrameRate": default_profile0["fps"]}
            )
            picam0.configure(config)
            stream_output0 = StreamingOutput(cam_id=0)
            stream_encoder = JpegEncoder(q=70)
            picam0.start_recording(stream_encoder, FileOutput(stream_output0), name="lores")
            video_encoder0 = H264Encoder(bitrate=10000000)
            picam0.start()
            reset_metrics(0)
            camera_settings[0] = {
                "profileId": default_profile0["id"],
                "resolution": list(default_profile0["main"]),
                "fps": default_profile0["fps"]
            }
            logging.info("Camera 0 (CVHAD) initialized and streaming.")
        except Exception as e:
            logging.error(f"Failed to initialize camera 0: {e}")
            picam0 = None

    # --- Initialize Camera 1 (IR Camera) ---
    if len(camera_ids) >= 2:
        try:
            logging.info("Initializing camera 1 (IR Camera)...")
            default_profile1 = next((p for p in CAMERA_PROFILES[1] if p["id"] == "cm3_720p25"), CAMERA_PROFILES[1][0])
            picam1 = Picamera2(camera_num=1)
            config = picam1.create_video_configuration(
                main={"size": default_profile1["main"]},
                lores={"size": default_profile1["lores"]},
                controls={"FrameRate": default_profile1["fps"]}
            )
            picam1.configure(config)
            stream_output1 = StreamingOutput(cam_id=1)
            stream_encoder = JpegEncoder(q=70)
            picam1.start_recording(stream_encoder, FileOutput(stream_output1), name="lores")
            video_encoder1 = H264Encoder(bitrate=10000000)
            picam1.start()
            reset_metrics(1)
            camera_settings[1] = {
                "profileId": default_profile1["id"],
                "resolution": list(default_profile1["main"]),
                "fps": default_profile1["fps"]
            }
            logging.info("Camera 1 (IR Camera) initialized and streaming.")
        except Exception as e:
            logging.error(f"Failed to initialize camera 1: {e}")
            picam1 = None
    
    time.sleep(1) # Allow cameras to warm up

# --- Helper to create placeholder SVG ---
def get_placeholder_svg(width, height, text):
    """Generates an SVG placeholder image."""
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <rect width="100%" height="100%" fill="#333"/>
        <text x="50%" y="50%" fill="#FFF" font-size="24" font-family="sans-serif"
              text-anchor="middle" dy=".3em">{text}</text>
    </svg>
    """
    return io.BytesIO(svg.encode('utf-8'))

# --- Web Routes ---

@app.route('/')
def index():
    """Video streaming home page."""
    # We inline the HTML and JavaScript to keep this to a single file
    html_content = """
    <html>
    <head>
        <title>Pi Camera - Dual Stream & Record</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                min-height: 100vh; 
                margin: 0; 
                background-color: #f4f4f5; /* Lighter gray */
                padding: 20px;
                box-sizing: border-box;
            }
            h1 { 
                color: #18181b; /* Darker text */
                margin-top: 0;
            }
            .header-nav {
                width: 100%;
                max-width: 1700px; /* Max width for large screens */
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .header-nav a {
                font-size: 1rem;
                font-weight: 500;
                color: #fff;
                background-color: #2563eb; /* Blue */
                padding: 8px 16px;
                border-radius: 8px;
                text-decoration: none;
                transition: background-color 0.2s;
            }
            .header-nav a:hover {
                background-color: #1d4ed8; /* Darker blue */
            }
            .container {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 25px;
                width: 100%;
                max-width: 1600px;
                margin: 0 auto;
            }
            .stream-box {
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #fff;
                border-radius: 12px;
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
                padding: 20px;
                border: 1px solid #e4e4e7;
                width: 100%;
                max-width: 100%;
                box-sizing: border-box;
            }
            h2 {
                font-size: 1.5rem; /* Larger title */font-weight: 600;
                color: #3f3f46;
                margin: 0 0 15px 0;
            }
            img { 
                background-color: #000;
                border: 1px solid #d4d4d8;
                border-radius: 8px;
                width: 100% !important;
                height: auto;
                aspect-ratio: 4 / 3;
            }
            .controls {
                margin-top: 20px;
                display: flex;
                gap: 12px;
                align-items: center;
            }
            .status {
                font-size: 1rem;
                font-weight: 500;
                padding: 8px 12px;
                border-radius: 6px;
                width: 150px;
                text-align: center;
            }
            .status.idle { background-color: #f4f4f5; color: #52525b; }
            .status.recording { background-color: #fee2e2; color: #dc2626; font-weight: 600; }
            
            .btn {
                font-size: 1rem;
                font-weight: 500;
                padding: 10px 18px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.2s, opacity 0.2s;
            }
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .btn-start { background-color: #22c55e; color: #fff; }
            .btn-start:hover:not(:disabled) { background-color: #16a34a; }
            .btn-stop { background-color: #ef4444; color: #fff; }
            .btn-stop:hover:not(:disabled) { background-color: #dc2626; }
            .settings {
                margin-top: 18px;
                width: 100%;
                display: flex;
                flex-direction: column;
                gap: 6px;
            }
            .settings label {
                font-size: 0.95rem;
                font-weight: 500;
                color: #3f3f46;
            }
            .settings select {
                padding: 8px 12px;
                border: 1px solid #d4d4d8;
                border-radius: 8px;
                font-size: 0.95rem;
                background-color: #fff;
            }
            .settings-status {
                font-size: 0.85rem;
                color: #2563eb;
                min-height: 1em;
            }
            .metrics-chart {
                width: 100%;
                height: 220px;
            }
            .metrics-chart canvas {
                width: 100% !important;
                height: 100% !important;
            }
        </style>
    </head>
    <body>
        <div class="header-nav">
            <h1>ERPL Specialty Camera Dashboard</h1>
            <a href="/recordings_list" target="_blank">View Recordings</a>
        </div>
        <div class="container">
            <!-- Camera 0 -->
            <div class="stream-box">
                <h2>CVHAD</h2>
                <img src="/video_feed_0" alt="CVHAD stream">
                <div class="metrics-chart">
                    <canvas id="metrics-chart-0"></canvas>
                </div>
                <div class="controls">
                    <button id="start-0" class="btn btn-start" onclick="startRecording(0)">Start Recording</button>
                    <button id="stop-0" class="btn btn-stop" onclick="stopRecording(0)">Stop Recording</button>
                    <span id="status-0" class="status idle">Idle</span>
                </div>
                <div class="settings">
                    <label for="profile-0">Resolution &amp; Frame Rate</label>
                    <select id="profile-0" onchange="handleProfileChange(0)" disabled>
                        <option>Loading…</option>
                    </select>
                    <div id="settings-status-0" class="settings-status"></div>
                </div>
            </div>
            <!-- Camera 1 -->
            <div class="stream-box">
                <h2>IR Camera</h2>
                <img src="/video_feed_1" alt="IR Camera stream">
                <div class="metrics-chart">
                    <canvas id="metrics-chart-1"></canvas>
                </div>
                <div class="controls">
                    <button id="start-1" class="btn btn-start" onclick="startRecording(1)">Start Recording</button>
                    <button id="stop-1" class="btn btn-stop" onclick="stopRecording(1)">Stop Recording</button>
                    <span id="status-1" class="status idle">Idle</span>
                </div>
                <div class="settings">
                    <label for="profile-1">Resolution &amp; Frame Rate</label>
                    <select id="profile-1" onchange="handleProfileChange(1)" disabled>
                        <option>Loading…</option>
                    </select>
                    <div id="settings-status-1" class="settings-status"></div>
                </div>
            </div>
        </div>

        <script>
            // On page load, fetch the current status of the cameras
            let cameraProfiles = {};
            let currentProfiles = {};
            let charts = {};
            let metricsTimer;
            window.onload = () => {
                getStatus();
                loadCameraSettings();
                initCharts();
            };

            function loadCameraSettings() {
                fetch('/settings/options')
                    .then(response => response.json())
                    .then(data => {
                        cameraProfiles = data.profiles || {};
                        currentProfiles = data.current || {};
                        [0, 1].forEach(populateProfileSelect);
                    })
                    .catch(err => console.error('Error loading camera settings:', err));
            }

            function populateProfileSelect(camNum) {
                const camKey = String(camNum);
                const select = document.getElementById(`profile-${camNum}`);
                const statusEl = document.getElementById(`settings-status-${camNum}`);
                if (!select || !statusEl) return;

                const profiles = (cameraProfiles && cameraProfiles[camKey]) || [];
                select.innerHTML = '';

                if (!profiles.length) {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'Camera unavailable';
                    select.appendChild(option);
                    select.disabled = true;
                    statusEl.textContent = '';
                    return;
                }

                profiles.forEach(profile => {
                    const option = document.createElement('option');
                    option.value = profile.id;
                    option.textContent = `${profile.resolution} @ ${profile.fps} fps`;
                    const current = currentProfiles ? currentProfiles[camKey] : null;
                    if (current && current.profileId === profile.id) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });

                select.disabled = false;
                statusEl.textContent = '';
            }

            function getStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Status received:', data);
                        updateUI(0, data.cam0);
                        updateUI(1, data.cam1);
                    })
                    .catch(err => console.error('Error fetching status:', err));
            }

            function updateUI(camNum, state) {
                const startBtn = document.getElementById(`start-${camNum}`);
                const stopBtn = document.getElementById(`stop-${camNum}`);
                const statusEl = document.getElementById(`status-${camNum}`);

                if (state.recording) {
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    statusEl.textContent = 'Recording...';
                    statusEl.className = 'status recording'; } else {
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    statusEl.textContent = 'Idle';
                    statusEl.className = 'status idle';
                }
            }

            function startRecording(camNum) {
                setControlsDisabled(camNum, true);
                fetch('/record/start/' + camNum, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        console.log(`Cam ${camNum} start:`, data);
                        if (data.error) {
                            alert(`Error: ${data.error}`);
                        }
                        getStatus(); // Refresh status from server
                    })
                    .catch(err => {
                        console.error('Error starting recording:', err);
                        alert('Error starting recording. See console.');
                        setControlsDisabled(camNum, false);
                    });
            }

            function stopRecording(camNum) {
                setControlsDisabled(camNum, true);
                fetch('/record/stop/' + camNum, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        console.log(`Cam ${camNum} stop:`, data);
                        if (data.error) {
                            alert(`Error: ${data.error}`);
                        }
                        getStatus(); // Refresh status from server
                    })
                    .catch(err => {
                        console.error('Error stopping recording:', err);
                        alert('Error stopping recording. See console.');
                        setControlsDisabled(camNum, false);
                    });
            }

            function handleProfileChange(camNum) {
                const select = document.getElementById(`profile-${camNum}`);
                const statusEl = document.getElementById(`settings-status-${camNum}`);
                if (!select || !select.value) return;

                const profileId = select.value;
                const camKey = String(camNum);

                statusEl.textContent = 'Applying...';
                select.disabled = true;
                setControlsDisabled(camNum, true);

                fetch('/settings/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cam: camNum, profileId })
                })
                    .then(response => response.json().then(data => ({ ok: response.ok, data })))
                    .then(result => {
                        if (!result.ok) {
                            throw new Error(result.data.error || 'Failed to apply settings');
                        }
                        currentProfiles[camKey] = result.data.current;
                        statusEl.textContent = 'Settings updated';
                        setTimeout(() => { statusEl.textContent = ''; }, 3000);
                        getStatus();
                    })
                    .catch(err => {
                        statusEl.textContent = err.message;
                        console.error('Error applying profile:', err);
                    })
                    .finally(() => {
                        select.disabled = false;
                        setControlsDisabled(camNum, false);
                    });
            }

            function initCharts() {
                [0, 1].forEach(cam => {
                    const canvas = document.getElementById(`metrics-chart-${cam}`);
                    if (!canvas) return;
                    charts[cam] = new Chart(canvas, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [
                                {
                                    label: 'Data rate (KB/s)',
                                    data: [],
                                    borderColor: '#2563eb',
                                    backgroundColor: 'rgba(37, 99, 235, 0.25)',
                                    borderWidth: 2,
                                    tension: 0.25,
                                    yAxisID: 'y'
                                },
                                {
                                    label: 'Estimated FPS',
                                    data: [],
                                    borderColor: '#dc2626',
                                    backgroundColor: 'rgba(220, 38, 38, 0.25)',
                                    borderWidth: 2,
                                    tension: 0.25,
                                    yAxisID: 'y1'
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            animation: false,
                            interaction: { mode: 'index', intersect: false },
                            plugins: {
                                legend: { labels: { color: '#1f2937' } },
                                tooltip: { mode: 'index', intersect: false }
                            },
                            scales: {
                                x: {
                                    ticks: { color: '#52525b', maxTicksLimit: 6 },
                                    grid: { color: 'rgba(79, 70, 229, 0.1)' }
                                },
                                y: {
                                    position: 'left',
                                    title: { display: true, text: 'KB/s', color: '#2563eb' },
                                    ticks: { color: '#2563eb' },
                                    grid: { color: 'rgba(37, 99, 235, 0.1)' }
                                },
                                y1: {
                                    position: 'right',
                                    title: { display: true, text: 'FPS', color: '#dc2626' },
                                    ticks: { color: '#dc2626' },
                                    grid: { drawOnChartArea: false }
                                }
                            }
                        }
                    });
                });
                fetchMetrics();
                metricsTimer = setInterval(fetchMetrics, 2000);
                window.addEventListener('beforeunload', () => clearInterval(metricsTimer));
            }

            function fetchMetrics() {
                fetch('/metrics')
                    .then(response => response.json())
                    .then(data => {
                        const metrics = data.metrics || {};
                        const fallbackTs = data.timestamp || Date.now() / 1000;
                        [0, 1].forEach(cam => {
                            const metric = metrics[String(cam)];
                            if (!metric) return;
                            updateMetricChart(cam, metric, metric.timestamp || fallbackTs);
                        });
                    })
                    .catch(err => console.error('Error fetching metrics:', err));
            }

            function updateMetricChart(camNum, metric, ts) {
                const chart = charts[camNum];
                if (!chart) return;
                const label = new Date(ts * 1000).toLocaleTimeString();
                chart.data.labels.push(label);
                chart.data.datasets[0].data.push(Number((metric.kbps || 0).toFixed(2)));
                chart.data.datasets[1].data.push(Number((metric.fps || 0).toFixed(2)));
                if (chart.data.labels.length > 30) {
                    chart.data.labels.shift();
                    chart.data.datasets.forEach(ds => ds.data.shift());
                }
                chart.update('none');
            }

            function setControlsDisabled(camNum, disabled) {
                document.getElementById(`start-${camNum}`).disabled = disabled;
                document.getElementById(`stop-${camNum}`).disabled = disabled;
            }
        </script>
    </body>
    </html>
    """
    return html_content

# --- Generator and Route for Camera 0 (CVHAD) ---
def gen_frames_0():
    """A generator function that yields camera 0 frames."""
    while True:
        if not picam0 or not stream_output0:
            logging.warning("Camera 0 not initialized, sending placeholder.")
            placeholder = get_placeholder_svg(800, 600, "CVHAD Offline")
            while True:
                frame = placeholder.getvalue()
                yield (b'--frame\r\n'
                       b'Content-Type: image/svg+xml\r\n\r\n' + frame + b'\r\n')
                time.sleep(1)

        try:
            with stream_output0.condition:
                stream_output0.condition.wait()
                frame = stream_output0.frame
            if not frame:
                logging.warning("No frame available (Cam 0), skipping.")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error in frame generator (Cam 0): {e}")
            break
    logging.info("Client disconnected from frame generator (Cam 0).")

@app.route('/video_feed_0')
def video_feed_0():
    """Video streaming route for camera 0."""
    return Response(gen_frames_0(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Generator and Route for Camera 1 (IR Camera) ---
def gen_frames_1():
    """A generator function that yields camera 1 frames."""
    while True:
        if not picam1 or not stream_output1:
            logging.warning("Camera 1 not initialized, sending placeholder.")
            placeholder = get_placeholder_svg(800, 600, "IR Camera Offline")
            while True:
                frame = placeholder.getvalue()
                yield (b'--frame\r\n'
                       b'Content-Type: image/svg+xml\r\n\r\n' + frame + b'\r\n')
                time.sleep(1)

        try:
            with stream_output1.condition:
                stream_output1.condition.wait()
                frame = stream_output1.frame
            if not frame:
                logging.warning("No frame available (Cam 1), skipping.")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error in frame generator (Cam 1): {e}")
            break
    logging.info("Client disconnected from frame generator (Cam 1).")

@app.route('/video_feed_1')
def video_feed_1():
    """Video streaming route for camera 1."""
    return Response(gen_frames_1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Recording API Routes ---

@app.route('/status', methods=['GET'])
def get_status():
    """Returns the current recording status of both cameras."""
    return jsonify({"cam0": cam0_state, "cam1": cam1_state})

@app.route('/record/start/<int:cam_num>', methods=['POST'])
def start_recording(cam_num):
    """Starts recording for the specified camera."""
    if cam_num == 0:
        lock, picam, encoder, state = cam0_lock, picam0, video_encoder0, cam0_state
        cam_name = "CVHAD"
    elif cam_num == 1:
        lock, picam, encoder, state = cam1_lock, picam1, video_encoder1, cam1_state
        cam_name = "IR Camera"
    else:
        return jsonify({"error": "Invalid camera number"}), 400

    with lock:
        if not picam or not encoder:
            logging.warning(f"Start record failed: Cam {cam_num} not initialized.")
            return jsonify({"error": f"{cam_name} is not connected or initialized"}), 404
        
        if state["recording"]:
            logging.warning(f"Start record failed: Cam {cam_num} already recording.")
            return jsonify({"error": f"{cam_name} is already recording"}), 400
        
        try:
            datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cam{cam_num}_{cam_name.lower().replace(' ', '')}_{datestr}.mp4"
            filepath = os.path.join(RECORDINGS_DIR, filename)
            
            # Start the H.264 encoder on the "main" stream
            picam.start_encoder(encoder, FileOutput(filepath), name="main")
            
            state["recording"] = True
            state["filename"] = filename
            logging.info(f"Started recording for Cam {cam_num} to {filename}")
            return jsonify({"message": "Recording started", "filename": filename})
        except Exception as e:
            logging.error(f"Failed to start recording for Cam {cam_num}: {e}")
            return jsonify({"error": f"Failed to start recording: {e}"}), 500

@app.route('/record/stop/<int:cam_num>', methods=['POST'])
def stop_recording(cam_num):
    """Stops recording for the specified camera."""
    if cam_num == 0:
        lock, picam, encoder, state = cam0_lock, picam0, video_encoder0, cam0_state
        cam_name = "CVHAD"
    elif cam_num == 1:
        lock, picam, encoder, state = cam1_lock, picam1, video_encoder1, cam1_state
        cam_name = "IR Camera"
    else:
        return jsonify({"error": "Invalid camera number"}), 400

    with lock:
        if not picam:
            logging.warning(f"Stop record failed: Cam {cam_num} not initialized.")
            return jsonify({"error": f"{cam_name} is not initialized"}), 404

        if not state["recording"]:
            logging.warning(f"Stop record failed: Cam {cam_num} is not recording.")
            return jsonify({"error": f"{cam_name} is not recording"}), 400
            
        try:
            # Stop the H.264 encoder
            picam.stop_encoder(encoder)
            
            filename = state["filename"]
            state["recording"] = False
            state["filename"] = None
            logging.info(f"Stopped recording for Cam {cam_num}. File saved: {filename}")
            return jsonify({"message": "Recording stopped", "filename": filename})
        except Exception as e:
            logging.error(f"Failed to stop recording for Cam {cam_num}: {e}")
            return jsonify({"error": f"Failed to stop recording: {e}"}), 500
# --- Download Routes ---

@app.route('/recordings_list')
def list_recordings():
    """Displays a list of available recordings for download."""
    try:
        files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.mp4')]
        files.sort(reverse=True) # Show newest first
    except Exception as e:
        logging.error(f"Could not list recordings directory: {e}")
        return f"<h1>Error listing recordings</h1><p>{e}</p>", 500

    # Simple HTML for the list
    list_html = """
    <style>
        body { font-family: sans-serif; padding: 20px; background-color: #f4f4f5; }
        h1 { color: #18181b; }
        a { 
            display: inline-block;
            margin: 5px 0;
            padding: 10px 15px;
            background-color: #fff;
            color: #2563eb;
            text-decoration: none;
            border-radius: 8px;
            border: 1px solid #e4e4e7;
            transition: background-color 0.2s;
            font-weight: 500;
        }
        a:hover { background-color: #fafafa; }
        p { color: #52525b; }
    </style>
    <h1>Available Recordings</h1>
    """
    if not files:
        list_html += "<p>No recordings found.</p>"
    else:
        for f in files:
            list_html += f'<a href="/download/{f}">{f}</a><br>'
    
    return list_html

@app.route('/download/<path:filename>')
def download_file(filename):
    """Serves a recording file for download."""
    # Basic security check
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
        
    try:
        return send_from_directory(RECORDINGS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"Download failed: File not found {filename}")
        return "File not found", 404
    except Exception as e:
        logging.error(f"Download failed: {e}")
        return "Error downloading file", 500

# --- Settings API Routes ---

@app.route('/settings/options', methods=['GET'])
def settings_options():
    profiles_payload = {
        str(cam): [
            {
                "id": profile["id"],
                "label": profile["label"],
                "resolution": f"{profile['main'][0]} x {profile['main'][1]}",
                "fps": profile["fps"]
            }
            for profile in CAMERA_PROFILES[cam]
        ]
        for cam in CAMERA_PROFILES
    }
    current_payload = {
        str(cam): camera_settings[cam] for cam in camera_settings
    }
    return jsonify({"profiles": profiles_payload, "current": current_payload})

@app.route('/settings/update', methods=['POST'])
def update_settings():
    data = request.get_json(force=True) or {}
    try:
        cam_num = int(data.get("cam"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid camera value"}), 400

    profile_id = data.get("profileId")
    if cam_num not in CAMERA_PROFILES:
        return jsonify({"error": "Camera not supported"}), 400

    profile = next((p for p in CAMERA_PROFILES[cam_num] if p["id"] == profile_id), None)
    if not profile:
        return jsonify({"error": "Profile not recognized"}), 400

    lock = cam0_lock if cam_num == 0 else cam1_lock
    state = cam0_state if cam_num == 0 else cam1_state
    picam = picam0 if cam_num == 0 else picam1
    stream_output = stream_output0 if cam_num == 0 else stream_output1

    if not picam or not stream_output:
        return jsonify({"error": "Camera is not initialized"}), 404

    with lock:
        if state["recording"]:
            return jsonify({"error": "Stop recording before changing settings"}), 400
        try:
            try:
                picam.stop_recording()
            except Exception as e:
                logging.warning(f"Ignored error stopping recording during profile update: {e}")
            try:
                picam.stop()
            except Exception as e:
                logging.warning(f"Ignored error stopping camera during profile update: {e}")

            config = picam.create_video_configuration(
                main={"size": profile["main"]},
                lores={"size": profile.get("lores", (800, 600))},
                controls={"FrameRate": profile["fps"]}
            )
            picam.configure(config)
            stream_output.frame = None
            stream_encoder = JpegEncoder(q=70)
            picam.start_recording(stream_encoder, FileOutput(stream_output), name="lores")
            picam.start();
            time.sleep(0.2);
            reset_metrics(cam_num);
        except Exception as e:
            logging.error(f"Failed to update camera {cam_num} profile: {e}")
            return jsonify({"error": f"Failed to apply profile: {e}"}), 500

        camera_settings[cam_num] = {
            "profileId": profile["id"],
            "resolution": list(profile["main"]),
            "fps": profile["fps"]
        }

    return jsonify({"message": "Settings updated", "current": camera_settings[cam_num]})

@app.route('/metrics', methods=['GET'])
def metrics():
    now = time.time()
    with metrics_lock:
        payload = {
            str(cam): {
                "kbps": float(state["bytes_per_sec"] / 1024) if state["bytes_per_sec"] else 0.0,
                "fps": float(state["fps"]),
                "timestamp": float(state["last_timestamp"] or now)
            }
            for cam, state in metrics_state.items()
        }
    return jsonify({"metrics": payload, "timestamp": now})

# --- Cleanup ---
def cleanup():
    """Stops cameras and encoders on script exit."""
    logging.info("Shutting down... stopping cameras.")
    if picam0:
        try:
            if cam0_state["recording"]:
                picam0.stop_encoder(video_encoder0)
                logging.info("Stopped Cam 0 recording.")
            picam0.stop()
            logging.info("Camera 0 stopped.")
        except Exception as e:
            logging.error(f"Error stopping Cam 0: {e}")
    if picam1:
        try:
            if cam1_state["recording"]:
                picam1.stop_encoder(video_encoder1)
                logging.info("Stopped Cam 1 recording.")
            picam1.stop()
            logging.info("Camera 1 stopped.")
        except Exception as e:
            logging.error(f"Error stopping Cam 1: {e}")

# Register the cleanup function to be called on exit
atexit.register(cleanup)

# --- Main Execution ---
if __name__ == '__main__':
    initialize_cameras()
    
    # Start the Flask web server
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 8000))
    logging.info(f"Starting Flask server on http://{host}:{port}")
    # 'use_reloader=False' is important to prevent re-initialization on save
    # 'threaded=True' is essential for handling multiple clients and streams
    app.run(host=host, port=port, threaded=True, use_reloader=False)