#!/usr/bin/python3
#
# This script creates a simple web server that streams video from a
# Raspberry Pi Camera Module to a web page in 1080p with color correction.
#
# It uses the picamera2 library, which is the officially supported
# library for Raspberry Pi cameras on recent versions of Raspberry Pi OS.
#
# --- SETUP INSTRUCTIONS ---
#
# 1. Install required packages using apt:
#    sudo apt update
#    sudo apt install python3-flask python3-picamera2 -y
#
# 2. Run the script from your terminal:
#    python3 camera_stream.py
#
# 3. Open a web browser on any device in the same network and navigate to:
#    http://<YOUR_RASPBERRY_PI_IP_ADDRESS>:8000
#
#    (You can find your Pi's IP address by running: hostname -I)
#

import io
import time
from threading import Condition

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, Quality
from picamera2.outputs import FileOutput

# --- HTML Page Template ---
# This is the HTML for the webpage that will display the video stream.
# It contains a single image element that points to the /video_feed route.
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raspberry Pi - NoIR Camera Stream (1080p)</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #2c3e50;
            color: #ecf0f1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        h1 {
            color: #3498db;
            margin-bottom: 20px;
        }
        .video-container {
            border: 3px solid #3498db;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            background-color: #000;
            width: 90%;
            max-width: 1200px; /* Increased max-width for 1080p */
        }
        img {
            display: block;
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>Raspberry Pi - NoIR Camera Stream (1080p)</h1>
    <div class="video-container">
        <!-- The 'src' of this image is the streaming endpoint -->
        <img src="{{ url_for('video_feed') }}">
    </div>
</body>
</html>
"""

# --- Streaming Output Class ---
# This class handles the camera's output stream. It's designed to be thread-safe
# and allows multiple clients to read the latest frame.
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Initialize the Flask web application
app = Flask(__name__)

# Initialize the Raspberry Pi camera
picam2 = Picamera2()
# Configure the camera for video, setting a 1080p resolution.
picam2.configure(picam2.create_video_configuration(main={"size": (1920, 1080)}))

time.sleep(1) # Allow camera to warm up before starting the stream.

output = StreamingOutput()
# Start recording to our custom streaming output, using a JPEG encoder.
# For 1080p, we can use a slightly higher quality.
picam2.start_recording(JpegEncoder(), FileOutput(output), quality=Quality.HIGH)

@app.route('/')
def index():
    """Route for the main web page."""
    # Render the HTML template we defined above
    return render_template_string(HTML_PAGE)

def generate_frames():
    """A generator function that yields camera frames for the video feed."""
    while True:
        with output.condition:
            # Wait until a new frame is available
            output.condition.wait()
            frame = output.frame
        # Yield the frame in the multipart format that browsers expect
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for the video streaming feed."""
    # Returns a multipart response, which streams the frames from the generator
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the Flask web server
    # host='0.0.0.0' makes the server accessible from any device on the network
    # threaded=True allows the server to handle multiple requests simultaneously
    print("Starting 1080p camera streaming server...")
    print("Open your browser and go to http://<YOUR_PI_IP>:8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)  # nosec B104


