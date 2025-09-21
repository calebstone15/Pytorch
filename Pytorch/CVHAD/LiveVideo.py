import queue
import cv2
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import PIL.Image, PIL.ImageTk
import time

class CameraHandler:
    def __init__(self, app):
        self.app = app  # Reference to main app
        self.camera = None
        self.camera_id = 0  # Default camera ID
        self.available_cameras = []
        self.window = None
        
        # Threading
        self.is_running = False
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None
    
    def create_interface(self, window):
        """Create the live video interface"""
        self.window = window
        self.window.geometry("900x700")
        
        # Create widgets
        ttk.Label(window, text="Live Video", font=("Arial", 16)).pack(pady=10)
        
        # Camera selection frame
        camera_frame = ttk.Frame(window, padding=10)
        camera_frame.pack(fill=tk.X, padx=20)
        
        # Camera selection label and dropdown
        ttk.Label(camera_frame, text="Select Camera:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, 
                                       width=40, state="readonly")
        self.camera_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        # Video display frame
        self.display_frame = tk.Frame(window, bg="black")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label for displaying video
        self.video_label = tk.Label(self.display_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event
        self.display_frame.bind("<Configure>", self.on_resize)
        self.display_width = 0
        self.display_height = 0
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Scanning for cameras...")
        status_bar = ttk.Label(window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Close button
        close_btn = ttk.Button(window, text="Close", command=self.close_window)
        close_btn.pack(pady=10)
        
        # Force update to get initial sizes
        window.update()
        self.on_resize(None)  # Initial size
        
        # Scan for cameras when the interface is created
        self.scan_cameras()
        
        # Protocol for closing the window
        window.protocol("WM_DELETE_WINDOW", self.close_window)
    
    def on_resize(self, event):
        """Update display dimensions on resize"""
        self.display_width = self.display_frame.winfo_width()
        self.display_height = self.display_frame.winfo_height()
    
    def scan_cameras(self):
        """Scan for available camera devices"""
        self.available_cameras = []
        self.status_var.set("Scanning for cameras...")
        self.window.update_idletasks()
        
        # Create a list to store camera information
        camera_info_list = []
        
        # Try cameras with indices 0 to 9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ret, _ = cap.read()  # Try to read a frame
                
                if ret:
                    camera_info = {
                        'index': i,
                        'resolution': f"{width}x{height}",
                        'description': f"Camera {i} ({width}x{height})"
                    }
                    self.available_cameras.append(camera_info)
                    camera_info_list.append(camera_info['description'])
                
                cap.release()
        
        # Update combobox with found cameras
        if camera_info_list:
            self.camera_combo['values'] = camera_info_list
            self.camera_combo.current(0)  # Select first camera
            self.status_var.set(f"Found {len(camera_info_list)} cameras")
            
            # Auto-connect to first camera
            self.connect_selected_camera()
        else:
            self.camera_combo['values'] = ["No cameras found"]
            self.camera_combo.current(0)
            self.status_var.set("No cameras detected")
    
    def on_camera_selected(self, event):
        """Called when a new camera is selected in the dropdown"""
        # Connect to the selected camera
        self.connect_selected_camera()
    
    def connect_selected_camera(self):
        """Connect to the camera selected in the combobox"""
        # Stop any running camera first
        if self.is_running:
            self.stop_camera()
        
        # Get selected camera index
        selected = self.camera_combo.current()
        if selected >= 0 and selected < len(self.available_cameras):
            camera_id = self.available_cameras[selected]['index']
            
            # Connect to the camera
            self.status_var.set(f"Connecting to camera {camera_id}...")
            if self.connect_camera(camera_id):
                self.is_running = True
                self.stop_event.clear()
                
                # Start capture thread
                self.start_capture_thread()
                
                # Start the update loop
                self.update_camera_frame()
            else:
                messagebox.showerror("Error", f"Could not connect to camera {camera_id}.")
                self.status_var.set("Failed to connect to camera")
    
    def connect_camera(self, camera_id):
        """Connect to a camera device"""
        self.camera_id = camera_id
        
        # Release any existing camera
        self.release()
        
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera with ID {self.camera_id}")
                return False
            
            # Set high resolution if possible
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Test frame read
            ret, _ = self.camera.read()
            if not ret:
                print("Error: Could not read frame from camera")
                self.release()
                return False
                
            return True
        except Exception as e:
            print(f"Error connecting to camera: {str(e)}")
            self.release()
            return False
    
    def start_capture_thread(self):
        """Start the background thread for capturing frames"""
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
    
    def capture_loop(self):
        """Loop to capture frames in background"""
        while not self.stop_event.is_set():
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            else:
                time.sleep(0.01)
    
    def update_camera_frame(self):
        """Update the displayed frame from camera"""
        if not self.is_running or self.stop_event.is_set():
            return
            
        try:
            frame = self.frame_queue.get_nowait()
            self.display_frame_on_canvas(frame)
            self.status_var.set(f"Connected: {self.get_camera_info()}")
        except queue.Empty:
            pass
            
        # Schedule the next update
        self.window.after(30, self.update_camera_frame)  # ~33ms for ~30fps
    
    def get_camera_info(self):
        """Get information about the current camera"""
        if self.camera is None or not self.camera.isOpened():
            return "No camera connected"
        
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        return f"Camera {self.camera_id} ({width}x{height}, {fps:.1f} fps)"
    
    def display_frame_on_canvas(self, frame):
        """Display frame on canvas"""
        if frame is None:
            return
        
        # Resize frame to fit display if necessary
        if self.display_width > 1 and self.display_height > 1:
            # Calculate aspect ratio
            img_height, img_width = frame.shape[:2]
            aspect_ratio = img_width / img_height
            
            # Determine new dimensions while maintaining aspect ratio
            if self.display_width / self.display_height > aspect_ratio:
                # Display is wider than the image
                new_height = self.display_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Display is taller than the image
                new_width = self.display_width
                new_height = int(new_width / aspect_ratio)
                
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert frame from BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        
        # Update image in label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def stop_camera(self):
        """Stop the camera stream"""
        self.stop_event.set()
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)  # Increased timeout for safety
            self.capture_thread = None
        self.release()
        while not self.frame_queue.empty():
            self.frame_queue.get()  # Clear queue
        self.status_var.set("Camera stopped")
    
    def release(self):
        """Release the camera resource"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def close_window(self):
        """Close the window and release resources"""
        self.stop_camera()
        if self.window:
            self.window.destroy()
            self.window = None

# For testing the module directly
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Live Video Test")
    root.geometry("800x600")
    
    app = type('', (), {})()  # Simple mock object
    camera_handler = CameraHandler(app)
    camera_handler.create_interface(root)
    
    root.mainloop()