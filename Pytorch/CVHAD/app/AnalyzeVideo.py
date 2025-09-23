import cv2
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import PIL.Image, PIL.ImageTk
import time

class VideoAnalyzer:
    def __init__(self, app):
        self.app = app  # Reference to main app
        self.video = None
        self.file_path = None
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 0
        self.duration = 0
        self.window = None
        
        # Threading and playback control
        self.is_running = False
        self.is_paused = False
        self.stop_event = threading.Event()
        self.slider_dragging = False
    
    def create_interface(self, window):
        """Create the video analysis interface"""
        self.window = window
        self.window.geometry("1200x1200")  # Larger window size
        
        # Create widgets
        ttk.Label(window, text="Analyze Video", font=("Arial", 16)).pack(pady=10)
        
        # File selection frame
        file_frame = ttk.Frame(window, padding=10)
        file_frame.pack(fill=tk.X, padx=20)
        
        # Load video button
        load_btn = ttk.Button(file_frame, text="Load Video File", command=self.load_video)
        load_btn.pack(pady=5)
        
        # Current file display
        self.file_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_var, wraplength=600)
        file_label.pack(pady=5)
        
        # Video display frame
        self.display_frame = tk.Frame(window, bg="black")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label for displaying video
        self.video_label = tk.Label(self.display_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Playback controls frame - Pack BEFORE display frame
        control_frame = ttk.Frame(window, padding=5)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Time display
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        time_label = ttk.Label(control_frame, textvariable=self.time_var, width=15)
        time_label.pack(side=tk.LEFT, padx=5)
        
        # Progress slider
        self.position_var = tk.DoubleVar(value=0)
        self.progress_slider = ttk.Scale(
            control_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.position_var,
            command=self.on_slider_change
        )
        self.progress_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        # Playback buttons frame
        button_frame = ttk.Frame(window, padding=5)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Play/Pause button
        self.play_pause_var = tk.StringVar(value="Play")
        self.play_pause_btn = ttk.Button(
            button_frame, 
            textvariable=self.play_pause_var,
            command=self.toggle_play_pause
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Remove Stop button
        # stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_video)
        # stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Restart button
        restart_btn = ttk.Button(button_frame, text="Restart", command=self.restart_video)
        restart_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load a video file to begin")
        status_bar = ttk.Label(window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Close button
        close_btn = ttk.Button(window, text="Close", command=self.close_window)
        close_btn.pack(pady=10)
        
        # Protocol for closing the window
        window.protocol("WM_DELETE_WINDOW", self.close_window)
    
    def load_video(self):
        """Open file dialog and load a video file"""
        # Stop any running video
        if self.is_running:
            self.stop_video()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:  # User canceled
            return
        
        self.status_var.set(f"Loading video: {os.path.basename(file_path)}")
        if self.open_video(file_path):
            self.file_var.set(f"File: {file_path}")
            # Set the video to pause initially
            self.is_running = True
            self.is_paused = True
            self.play_pause_var.set("Play")
            self.stop_event.clear()
            
            # Show first frame
            self.seek_to_frame(0)
            self.update_time_display()
        else:
            messagebox.showerror("Error", "Could not open the video file.")
            self.status_var.set("Failed to load video")
    
    def open_video(self, file_path):
        """Load a video file"""
        if not os.path.exists(file_path):
            return False
        
        # Release any previously loaded video
        self.release()
        
        try:
            self.video = cv2.VideoCapture(file_path)
            
            if not self.video.isOpened():
                return False
            
            # Get video properties
            self.file_path = file_path
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.current_frame = 0
            
            # Calculate duration in seconds
            if self.fps > 0:
                self.duration = self.frame_count / self.fps
            else:
                self.duration = 0
            
            return True
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            self.release()
            return False
    
    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if not self.is_running:
            return
            
        self.is_paused = not self.is_paused
        self.play_pause_var.set("Pause" if not self.is_paused else "Play")
        
        if not self.is_paused:
            # Start playback
            self.update_video_frame()
    
    def restart_video(self):
        """Restart the video from the beginning"""
        if not self.is_running:
            return
            
        self.seek_to_frame(0)
        
        # If the video was playing, continue playing from the start
        if not self.is_paused:
            self.update_video_frame()
    
    def on_slider_press(self, event):
        """Handle slider press event"""
        self.slider_dragging = True
    
    def on_slider_release(self, event):
        """Handle slider release event"""
        if self.slider_dragging:
            position = self.position_var.get()
            self.seek_to_position(position)
            self.slider_dragging = False
            
            # If video was playing, continue playing from new position
            if not self.is_paused:
                self.update_video_frame()
    
    def on_slider_change(self, value):
        """Handle slider change event"""
        if self.slider_dragging and self.frame_count > 0:
            # Update time display while dragging
            position = float(value)
            frame_num = int((position / 100) * self.frame_count)
            time_sec = frame_num / self.fps if self.fps > 0 else 0
            time_str = self.format_time(time_sec)
            total_time = self.format_time(self.duration)
            self.time_var.set(f"{time_str} / {total_time}")
    
    def seek_to_position(self, position_percent):
        """Seek to a position in the video based on percentage"""
        if not self.is_running or self.frame_count <= 0:
            return False
        
        # Calculate frame number
        frame_number = int((position_percent / 100) * self.frame_count)
        return self.seek_to_frame(frame_number)
    
    def seek_to_frame(self, frame_number):
        """Seek to a specific frame in the video"""
        if self.video is None or not self.video.isOpened():
            return False
        
        # Ensure frame number is within bounds
        frame_number = max(0, min(frame_number, self.frame_count - 1))
        
        # Set position
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        
        # Read and display the frame
        ret, frame = self.video.read()
        if ret:
            self.display_frame_on_canvas(frame)
            self.update_time_display()
            return True
        return False
    
    def update_video_frame(self):
        """Update the displayed frame from video"""
        if not self.is_running or self.is_paused or self.stop_event.is_set():
            return
            
        frame = self.get_frame()
        if frame is not None:
            self.display_frame_on_canvas(frame)
            
            # Update time and slider
            if not self.slider_dragging:
                self.update_time_display()
                # Update slider position
                if self.frame_count > 0:
                    position = (self.current_frame / self.frame_count) * 100
                    self.position_var.set(position)
            
            # Schedule next frame
            delay_ms = int(1000 / self.fps) if self.fps > 0 else 33  # ~30fps default
            self.window.after(delay_ms, self.update_video_frame)
        else:
            # End of video
            self.is_paused = True
            self.play_pause_var.set("Play")
            self.status_var.set("End of video")
    
    def get_frame(self):
        """Get the next frame from the video"""
        if self.video is None or not self.video.isOpened():
            return None
        
        if self.current_frame >= self.frame_count:
            return None
        
        ret, frame = self.video.read()
        if not ret:
            return None
        
        self.current_frame += 1
        return frame
    
    def update_time_display(self):
        """Update the time display"""
        if self.fps <= 0 or not self.is_running:
            return
            
        current_seconds = self.current_frame / self.fps
        current_time = self.format_time(current_seconds)
        total_time = self.format_time(self.duration)
        self.time_var.set(f"{current_time} / {total_time}")
        
        # Update status bar
        filename = os.path.basename(self.file_path) if self.file_path else "No file"
        percent = int((self.current_frame / self.frame_count) * 100) if self.frame_count > 0 else 0
        self.status_var.set(f"{filename} - {current_time}/{total_time} ({percent}%)")
    
    def format_time(self, seconds):
        """Format seconds as MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def display_frame_on_canvas(self, frame):
        """Display frame on canvas"""
        if frame is None:
            return
        
        # Resize frame to fit display
        display_width = self.display_frame.winfo_width()
        display_height = self.display_frame.winfo_height()
        
        if display_width > 1 and display_height > 1:
            # Calculate aspect ratio
            img_height, img_width = frame.shape[:2]
            aspect_ratio = img_width / img_height
            
            # Determine new dimensions while maintaining aspect ratio
            if display_width / display_height > aspect_ratio:
                # Display is wider than the image
                new_height = display_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Display is taller than the image
                new_width = display_width
                new_height = int(new_width / aspect_ratio)
                
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert frame from BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        
        # Update image in label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def stop_video(self):
        """Stop video playback"""
        self.stop_event.set()
        self.is_running = False
        self.is_paused = True
        self.play_pause_var.set("Play")
        self.status_var.set("Video stopped")
        self.position_var.set(0)
        self.time_var.set("00:00 / 00:00")
    
    def release(self):
        """Release the video resource"""
        if self.video is not None:
            self.video.release()
            self.video = None
            self.file_path = None
            self.frame_count = 0
            self.current_frame = 0
            self.fps = 0
            self.duration = 0
    
    def close_window(self):
        """Close the window and release resources"""
        self.stop_video()
        self.release()
        if self.window:
            self.window.destroy()
            self.window = None

# For testing the module directly
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Analyze Video Test")
    root.geometry("800x600")
    
    app = type('', (), {})()  # Simple mock object
    video_analyzer = VideoAnalyzer(app)
    video_analyzer.create_interface(root)
    
    root.mainloop()
