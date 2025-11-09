import io
import logging  
import time
from http import server
from threading import Condition
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, Quality
from picamera2.outputs import FileOutput
import HotVizControlStyle  # Import the CSS file
import HotVizControlPanel

# --- HTML Page Template ---

HTML_PAGE = HotVizControlPanel

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

app = Flask(__name__)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1920, 1080)}))
time.sleep(1)  # Allow camera to warm up

output= StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output), quality=Quality.HIGH)

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_PAGE)

