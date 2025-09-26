import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading  # For clipboard functionality
import glob

def select_video():
    """Opens file dialog to select a video file."""
    file_path = filedialog.askopenfilename(
        title="SELECT INPUT VIDEO FILE",
        filetypes=[
            ("All files", "*.*"),
            ("M4V files", "*.m4v"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.mpg *.mpeg *.m4v"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("MKV files", "*.mkv"),
            ("WMV files", "*.wmv"),
            ("FLV files", "*.flv"),
            ("WebM files", "*.webm"),
            ("MPEG files", "*.mpg *.mpeg"),
        ]
    )
    return file_path

def select_output_directory():
    """Opens dialog to select output directory."""
    directory = filedialog.askdirectory(title="SELECT OUTPUT DIRECTORY FOR FRAMES")
    return directory

def process_video(video_path, output_dir, progress_callback=None):
    """Extract all frames from video and save as JPG files."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Error: Could not open video file."
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract frames
    count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if success:
            output_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            count += 1
            
            # Update progress
            if progress_callback and total_frames > 0:
                progress = (count / total_frames) * 100
                progress_callback(progress, count, total_frames)
    
    # Release video capture
    cap.release()
    return True, f"Extracted {count} frames to {output_dir}"

def images_to_video(image_folder, output_video_path, fps, progress_callback=None):
    """Create a video from a folder of images."""
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.jpeg')) + glob.glob(os.path.join(image_folder, '*.png')))
    if not image_files:
        return False, "No JPG images found in the specified folder."

    # Read first image to get frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    total_frames = len(image_files)
    for i, filename in enumerate(image_files):
        frame = cv2.imread(filename)
        out.write(frame)
        if progress_callback:
            progress = ((i + 1) / total_frames) * 100
            progress_callback(progress, i + 1, total_frames)

    out.release()
    return True, f"Successfully created video at {output_video_path}"

def is_valid_video(video_path):
    """Check if a file is a valid video that can be opened by OpenCV."""
    if not video_path or not os.path.exists(video_path):
        return False, "File does not exist."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False, "Could not open video file. Format may not be supported."
    
    # Check if we can read at least one frame
    ret, _ = cap.read()
    cap.release()
    
    if not ret:
        return False, "Could not read frames from the video."
    
    return True, "Valid video file."

def main():
    # Create window
    window = tk.Tk()
    window.title("Video/Image Frame Converter")
    window.geometry("1000x650")
    window.configure(padx=30, pady=30, bg="#23272e")
    
    # Try to make window prettier on macOS
    try:
        window.tk.call('::tk::unsupported::MacWindowStyle', 'style', window._w, 'document', 'modified')
    except:
        pass  # Ignore if not on macOS
    
    # --- Create Notebook for Tabs ---
    notebook = ttk.Notebook(window)
    notebook.pack(expand=True, fill='both')

    # --- Tab 1: Video to JPGs ---
    v2j_frame = ttk.Frame(notebook, padding=10)
    notebook.add(v2j_frame, text='Video to Frames')

    # --- Tab 2: JPGs to Video ---
    j2v_frame = ttk.Frame(notebook, padding=10)
    notebook.add(j2v_frame, text='Frames to Video')

    # ===================================================================
    # UI and Logic for Tab 1: Video to JPGs
    # ===================================================================

    # Status variables
    v2j_status_var = tk.StringVar()
    v2j_status_var.set("Ready - Select a video file to begin")
    v2j_progress_var = tk.DoubleVar()
    v2j_progress_var.set(0)
    
    # Path variables
    v2j_input_path_var = tk.StringVar()
    v2j_input_path_var.set("No video selected")
    v2j_output_path_var = tk.StringVar()
    v2j_output_path_var.set("No output directory selected")
    
    # Define functions for UI interaction
    def v2j_update_progress(progress, current, total):
        v2j_progress_var.set(progress)
        v2j_status_var.set(f"Processing: {current}/{total} frames ({progress:.1f}%)")
        window.update_idletasks()
    
    def validate_path(path, is_video=True):
        """Validates a path entered or pasted by the user"""
        if not path:
            return False, "No path provided"
        
        # Clean up the path (remove {} or extra quotes if present)
        if path.startswith('{') and path.endswith('}'): 
            path = path[1:-1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if path.startswith('[') and path.endswith(']'):
            path = path[1:-1]
        
        if is_video:
            if os.path.isfile(path):
                return is_valid_video(path)
            else:
                return False, "Not a valid file path"
        else:
            if os.path.isdir(path):
                return True, "Valid directory"
            else:
                return False, "Not a valid directory path"
    
    def v2j_select_input():
        path = select_video()
        if path:
            # Validate video before accepting it
            valid, message = is_valid_video(path)
            if valid:
                v2j_input_path_var.set(path)
                v2j_status_var.set("Video selected successfully")
                v2j_check_ready()
            else:
                messagebox.showerror("Invalid Video", message)
                v2j_status_var.set(message)
    
    def v2j_select_output():
        path = select_output_directory()
        if path:
            v2j_output_path_var.set(path)
            v2j_check_ready()
    
    def v2j_paste_input_path():
        try:
            path = pyperclip.paste()
            valid, message = validate_path(path, is_video=True)
            if valid:
                v2j_input_path_var.set(path)
                v2j_status_var.set("Video path pasted successfully")
                v2j_check_ready()
            else:
                messagebox.showerror("Invalid Video", message)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to paste: {str(e)}")
    
    def v2j_paste_output_path():
        try:
            path = pyperclip.paste()
            valid, message = validate_path(path, is_video=False)
            if valid:
                v2j_output_path_var.set(path)
                v2j_status_var.set("Output directory pasted successfully")
                v2j_check_ready()
            else:
                messagebox.showerror("Invalid Directory", message)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to paste: {str(e)}")
    
    def v2j_input_path_entry_update(event=None):
        """Process manual entry in the input path field"""
        path = v2j_input_path_entry.get()
        valid, message = validate_path(path, is_video=True)
        if valid:
            v2j_input_path_var.set(path)
            v2j_check_ready()
        else:
            v2j_input_path_var.set("Invalid path")
            v2j_status_var.set(message)
    
    def v2j_output_path_entry_update(event=None):
        """Process manual entry in the output path field"""
        path = v2j_output_path_entry.get()
        valid, message = validate_path(path, is_video=False)
        if valid:
            v2j_output_path_var.set(path)
            v2j_check_ready()
        else:
            v2j_output_path_var.set("Invalid path")
            v2j_status_var.set(message)
    
    def v2j_check_ready():
        # Enable process button only if both paths are selected
        if (v2j_input_path_var.get() != "No video selected" and 
            v2j_input_path_var.get() != "Invalid path" and
            v2j_output_path_var.get() != "No output directory selected" and
            v2j_output_path_var.get() != "Invalid path"):
            v2j_process_button.config(state=tk.NORMAL)
        else:
            v2j_process_button.config(state=tk.DISABLED)
    
    def v2j_start_processing():
        # Get paths from variables
        video_path = v2j_input_path_var.get()
        output_dir = v2j_output_path_var.get()
        
        # Update status
        v2j_status_var.set("Processing video...")
        v2j_progress_var.set(0)
        
        # Disable buttons during processing
        v2j_input_button.config(state=tk.DISABLED)
        v2j_paste_input_button.config(state=tk.DISABLED)
        v2j_input_path_entry.config(state=tk.DISABLED)
        v2j_output_button.config(state=tk.DISABLED)
        v2j_paste_output_button.config(state=tk.DISABLED)
        v2j_output_path_entry.config(state=tk.DISABLED)
        v2j_process_button.config(state=tk.DISABLED)
        
        # Run processing in a separate thread to keep GUI responsive
        def process_thread():
            success, message = process_video(video_path, output_dir, v2j_update_progress)
            
            # Re-enable buttons
            v2j_input_button.config(state=tk.NORMAL)
            v2j_paste_input_button.config(state=tk.NORMAL)
            v2j_input_path_entry.config(state=tk.NORMAL)
            v2j_output_button.config(state=tk.NORMAL)
            v2j_paste_output_button.config(state=tk.NORMAL)
            v2j_output_path_entry.config(state=tk.NORMAL)
            v2j_process_button.config(state=tk.NORMAL)
            
            v2j_status_var.set(message)
            
            if success:
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)
        
        threading.Thread(target=process_thread).start()
    
    # Create main frame
    main_frame = v2j_frame
    
    # Title
    title_label = ttk.Label(main_frame, 
                         text="Video to JPG Frame Extractor", 
                         font=("Arial", 18, "bold"))
    title_label.pack(pady=15)
    
    # Input section
    input_frame = ttk.LabelFrame(main_frame, text="Input Video", padding=10)
    input_frame.pack(fill=tk.X, pady=10, padx=5)
    
    input_controls = ttk.Frame(input_frame)
    input_controls.pack(fill=tk.X)
    
    # Row 1: Path entry and select button
    v2j_input_path_entry = ttk.Entry(input_controls, width=50)
    v2j_input_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    v2j_input_path_entry.bind("<FocusOut>", v2j_input_path_entry_update)
    v2j_input_path_entry.bind("<Return>", v2j_input_path_entry_update)
    
    v2j_input_button = ttk.Button(input_controls, text="Browse...", command=v2j_select_input)
    v2j_input_button.pack(side=tk.LEFT, padx=5)
    
    v2j_paste_input_button = ttk.Button(input_controls, text="Paste Path", command=v2j_paste_input_path)
    v2j_paste_input_button.pack(side=tk.LEFT, padx=5)
    
    # Row 2: Selected path
    input_path_display = ttk.Label(input_frame, textvariable=v2j_input_path_var, 
                               wraplength=700, justify=tk.LEFT)
    input_path_display.pack(fill=tk.X, pady=5)
    
    # Output section
    output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding=10)
    output_frame.pack(fill=tk.X, pady=10, padx=5)
    
    output_controls = ttk.Frame(output_frame)
    output_controls.pack(fill=tk.X)
    
    # Row 1: Path entry and select button
    v2j_output_path_entry = ttk.Entry(output_controls, width=50)
    v2j_output_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    v2j_output_path_entry.bind("<FocusOut>", v2j_output_path_entry_update)
    v2j_output_path_entry.bind("<Return>", v2j_output_path_entry_update)
    
    v2j_output_button = ttk.Button(output_controls, text="Browse...", command=v2j_select_output)
    v2j_output_button.pack(side=tk.LEFT, padx=5)
    
    v2j_paste_output_button = ttk.Button(output_controls, text="Paste Path", command=v2j_paste_output_path)
    v2j_paste_output_button.pack(side=tk.LEFT, padx=5)
    
    # Row 2: Selected path
    output_path_display = ttk.Label(output_frame, textvariable=v2j_output_path_var, 
                                wraplength=700, justify=tk.LEFT)
    output_path_display.pack(fill=tk.X, pady=5)
    
    # Process button
    v2j_process_button = ttk.Button(main_frame, text="PROCESS VIDEO TO FRAMES", 
                             command=v2j_start_processing, 
                             state=tk.DISABLED)
    v2j_process_button.pack(pady=20, fill=tk.X, padx=30)
    
    # Progress section
    progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
    progress_frame.pack(fill=tk.X, pady=10, padx=5)
    
    progress_bar = ttk.Progressbar(progress_frame, variable=v2j_progress_var, 
                                length=700, mode='determinate')
    progress_bar.pack(pady=10, fill=tk.X)
    
    status_label = ttk.Label(progress_frame, textvariable=v2j_status_var)
    status_label.pack(pady=5)
    
    # ===================================================================
    # UI and Logic for Tab 2: JPGs to Video
    # ===================================================================
    
    # Status variables
    j2v_status_var = tk.StringVar()
    j2v_status_var.set("Ready - Select an image folder to begin")
    j2v_progress_var = tk.DoubleVar()
    j2v_progress_var.set(0)
    
    # Path variables
    j2v_input_path_var = tk.StringVar()
    j2v_input_path_var.set("No image folder selected")
    j2v_output_path_var = tk.StringVar()
    j2v_output_path_var.set("No output video file selected")
    j2v_fps_var = tk.StringVar()
    j2v_fps_var.set("30")

    def j2v_update_progress(progress, current, total):
        j2v_progress_var.set(progress)
        j2v_status_var.set(f"Processing: {current}/{total} frames ({progress:.1f}%)")
        window.update_idletasks()

    def j2v_select_input_dir():
        path = filedialog.askdirectory(title="SELECT INPUT IMAGE FOLDER")
        if path:
            j2v_input_path_var.set(path)
            j2v_check_ready()

    def j2v_select_output_file():
        path = filedialog.asksaveasfilename(
            title="SELECT OUTPUT VIDEO FILE",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            defaultextension=".mp4"
        )
        if path:
            j2v_output_path_var.set(path)
            j2v_check_ready()

    def j2v_check_ready():
        try:
            fps = float(j2v_fps_var.get())
            fps_valid = fps > 0
        except ValueError:
            fps_valid = False

        if (os.path.isdir(j2v_input_path_var.get()) and
            j2v_output_path_var.get() != "No output video file selected" and
            fps_valid):
            j2v_process_button.config(state=tk.NORMAL)
        else:
            j2v_process_button.config(state=tk.DISABLED)

    def j2v_start_processing():
        image_folder = j2v_input_path_var.get()
        output_video = j2v_output_path_var.get()
        try:
            fps = float(j2v_fps_var.get())
            if fps <= 0:
                messagebox.showerror("Invalid FPS", "FPS must be a positive number.")
                return
        except ValueError:
            messagebox.showerror("Invalid FPS", "FPS must be a valid number.")
            return

        j2v_status_var.set("Processing images...")
        j2v_progress_var.set(0)
        
        # Disable buttons
        j2v_process_button.config(state=tk.DISABLED)

        def process_thread():
            success, message = images_to_video(image_folder, output_video, fps, j2v_update_progress)
            j2v_process_button.config(state=tk.NORMAL)
            j2v_status_var.set(message)
            if success:
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)

        threading.Thread(target=process_thread).start()

    # Main frame for tab 2
    main_frame_2 = j2v_frame

    title_label_2 = ttk.Label(main_frame_2, text="JPG Frames to Video Creator", font=("Arial", 18, "bold"))
    title_label_2.pack(pady=15)

    # Input image folder section
    input_frame_2 = ttk.LabelFrame(main_frame_2, text="Input Image Folder", padding=10)
    input_frame_2.pack(fill=tk.X, pady=10, padx=5)
    
    input_controls_2 = ttk.Frame(input_frame_2)
    input_controls_2.pack(fill=tk.X)
    
    j2v_input_path_entry = ttk.Entry(input_controls_2, textvariable=j2v_input_path_var, width=50)
    j2v_input_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    j2v_input_button = ttk.Button(input_controls_2, text="Browse...", command=j2v_select_input_dir)
    j2v_input_button.pack(side=tk.LEFT, padx=5)

    # Output video file section
    output_frame_2 = ttk.LabelFrame(main_frame_2, text="Output Video File", padding=10)
    output_frame_2.pack(fill=tk.X, pady=10, padx=5)
    
    output_controls_2 = ttk.Frame(output_frame_2)
    output_controls_2.pack(fill=tk.X)
    
    j2v_output_path_entry = ttk.Entry(output_controls_2, textvariable=j2v_output_path_var, width=50)
    j2v_output_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    j2v_output_button = ttk.Button(output_controls_2, text="Save As...", command=j2v_select_output_file)
    j2v_output_button.pack(side=tk.LEFT, padx=5)

    # Settings section
    settings_frame_2 = ttk.LabelFrame(main_frame_2, text="Settings", padding=10)
    settings_frame_2.pack(fill=tk.X, pady=10, padx=5)
    
    fps_label = ttk.Label(settings_frame_2, text="Frames Per Second (FPS):")
    fps_label.pack(side=tk.LEFT, padx=5)
    
    fps_entry = ttk.Entry(settings_frame_2, textvariable=j2v_fps_var, width=10)
    fps_entry.pack(side=tk.LEFT, padx=5)
    fps_entry.bind("<KeyRelease>", lambda e: j2v_check_ready())

    # Process button for tab 2
    j2v_process_button = ttk.Button(main_frame_2, text="PROCESS FRAMES TO VIDEO", command=j2v_start_processing, state=tk.DISABLED)
    j2v_process_button.pack(pady=20, fill=tk.X, padx=30)

    # Progress section for tab 2
    progress_frame_2 = ttk.LabelFrame(main_frame_2, text="Progress", padding=10)
    progress_frame_2.pack(fill=tk.X, pady=10, padx=5)
    
    progress_bar_2 = ttk.Progressbar(progress_frame_2, variable=j2v_progress_var, length=700, mode='determinate')
    progress_bar_2.pack(pady=10, fill=tk.X)
    
    status_label_2 = ttk.Label(progress_frame_2, textvariable=j2v_status_var)
    status_label_2.pack(pady=5)

    # Instructions
    instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding=10)
    instructions_frame.pack(fill=tk.X, pady=10, padx=5)
    
    instructions_text = """
    1. Select a video file using the Browse button or paste a file path.
    2. Select an output directory using the Browse button or paste a directory path.
    3. Click PROCESS VIDEO to extract all frames.
    
    Note: Large videos may take a long time to process.
    """
    instructions_label = ttk.Label(instructions_frame, text=instructions_text, 
                               wraplength=700, justify=tk.LEFT)
    instructions_label.pack(pady=5)
    
    # Initialize pyperclip - needed for paste functionality
    try:
        import pyperclip
    except ImportError:
        messagebox.showwarning("Missing Package", 
                             "The 'pyperclip' package is not installed. Paste functionality won't work.\n\n"
                             "To install it, run: pip install pyperclip")
    
    # Start main loop
    window.mainloop()

if __name__ == "__main__":
    main()