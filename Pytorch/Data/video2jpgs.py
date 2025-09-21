import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pyperclip  # For clipboard functionality

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
    window.title("Video to JPGs Converter")
    window.geometry("1000x600")
    window.configure(padx=30, pady=30, bg="#23272e")
    
    # Try to make window prettier on macOS
    try:
        window.tk.call('::tk::unsupported::MacWindowStyle', 'style', window._w, 'document', 'modified')
    except:
        pass  # Ignore if not on macOS
    
    # Status variables
    status_var = tk.StringVar()
    status_var.set("Ready - Select a video file to begin")
    progress_var = tk.DoubleVar()
    progress_var.set(0)
    
    # Path variables
    input_path_var = tk.StringVar()
    input_path_var.set("No video selected")
    output_path_var = tk.StringVar()
    output_path_var.set("No output directory selected")
    
    # Define functions for UI interaction
    def update_progress(progress, current, total):
        progress_var.set(progress)
        status_var.set(f"Processing: {current}/{total} frames ({progress:.1f}%)")
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
    
    def select_input():
        path = select_video()
        if path:
            # Validate video before accepting it
            valid, message = is_valid_video(path)
            if valid:
                input_path_var.set(path)
                status_var.set("Video selected successfully")
                check_ready()
            else:
                messagebox.showerror("Invalid Video", message)
                status_var.set(message)
    
    def select_output():
        path = select_output_directory()
        if path:
            output_path_var.set(path)
            check_ready()
    
    def paste_input_path():
        try:
            path = pyperclip.paste()
            valid, message = validate_path(path, is_video=True)
            if valid:
                input_path_var.set(path)
                status_var.set("Video path pasted successfully")
                check_ready()
            else:
                messagebox.showerror("Invalid Video", message)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to paste: {str(e)}")
    
    def paste_output_path():
        try:
            path = pyperclip.paste()
            valid, message = validate_path(path, is_video=False)
            if valid:
                output_path_var.set(path)
                status_var.set("Output directory pasted successfully")
                check_ready()
            else:
                messagebox.showerror("Invalid Directory", message)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to paste: {str(e)}")
    
    def input_path_entry_update(event=None):
        """Process manual entry in the input path field"""
        path = input_path_entry.get()
        valid, message = validate_path(path, is_video=True)
        if valid:
            input_path_var.set(path)
            check_ready()
        else:
            input_path_var.set("Invalid path")
            status_var.set(message)
    
    def output_path_entry_update(event=None):
        """Process manual entry in the output path field"""
        path = output_path_entry.get()
        valid, message = validate_path(path, is_video=False)
        if valid:
            output_path_var.set(path)
            check_ready()
        else:
            output_path_var.set("Invalid path")
            status_var.set(message)
    
    def check_ready():
        # Enable process button only if both paths are selected
        if (input_path_var.get() != "No video selected" and 
            input_path_var.get() != "Invalid path" and
            output_path_var.get() != "No output directory selected" and
            output_path_var.get() != "Invalid path"):
            process_button.config(state=tk.NORMAL)
        else:
            process_button.config(state=tk.DISABLED)
    
    def start_processing():
        # Get paths from variables
        video_path = input_path_var.get()
        output_dir = output_path_var.get()
        
        # Update status
        status_var.set("Processing video...")
        progress_var.set(0)
        
        # Disable buttons during processing
        input_button.config(state=tk.DISABLED)
        paste_input_button.config(state=tk.DISABLED)
        input_path_entry.config(state=tk.DISABLED)
        output_button.config(state=tk.DISABLED)
        paste_output_button.config(state=tk.DISABLED)
        output_path_entry.config(state=tk.DISABLED)
        process_button.config(state=tk.DISABLED)
        
        # Run processing in a separate thread to keep GUI responsive
        def process_thread():
            success, message = process_video(video_path, output_dir, update_progress)
            
            # Re-enable buttons
            input_button.config(state=tk.NORMAL)
            paste_input_button.config(state=tk.NORMAL)
            input_path_entry.config(state=tk.NORMAL)
            output_button.config(state=tk.NORMAL)
            paste_output_button.config(state=tk.NORMAL)
            output_path_entry.config(state=tk.NORMAL)
            process_button.config(state=tk.NORMAL)
            
            status_var.set(message)
            
            if success:
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)
        
        threading.Thread(target=process_thread).start()
    
    # Create main frame
    main_frame = ttk.Frame(window, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
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
    input_path_entry = ttk.Entry(input_controls, width=50)
    input_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    input_path_entry.bind("<FocusOut>", input_path_entry_update)
    input_path_entry.bind("<Return>", input_path_entry_update)
    
    input_button = ttk.Button(input_controls, text="Browse...", command=select_input)
    input_button.pack(side=tk.LEFT, padx=5)
    
    paste_input_button = ttk.Button(input_controls, text="Paste Path", command=paste_input_path)
    paste_input_button.pack(side=tk.LEFT, padx=5)
    
    # Row 2: Selected path
    input_path_display = ttk.Label(input_frame, textvariable=input_path_var, 
                               wraplength=700, justify=tk.LEFT)
    input_path_display.pack(fill=tk.X, pady=5)
    
    # Output section
    output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding=10)
    output_frame.pack(fill=tk.X, pady=10, padx=5)
    
    output_controls = ttk.Frame(output_frame)
    output_controls.pack(fill=tk.X)
    
    # Row 1: Path entry and select button
    output_path_entry = ttk.Entry(output_controls, width=50)
    output_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    output_path_entry.bind("<FocusOut>", output_path_entry_update)
    output_path_entry.bind("<Return>", output_path_entry_update)
    
    output_button = ttk.Button(output_controls, text="Browse...", command=select_output)
    output_button.pack(side=tk.LEFT, padx=5)
    
    paste_output_button = ttk.Button(output_controls, text="Paste Path", command=paste_output_path)
    paste_output_button.pack(side=tk.LEFT, padx=5)
    
    # Row 2: Selected path
    output_path_display = ttk.Label(output_frame, textvariable=output_path_var, 
                                wraplength=700, justify=tk.LEFT)
    output_path_display.pack(fill=tk.X, pady=5)
    
    # Process button
    process_button = ttk.Button(main_frame, text="PROCESS VIDEO", 
                             command=start_processing, 
                             state=tk.DISABLED)
    process_button.pack(pady=20, fill=tk.X, padx=30)
    
    # Progress section
    progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
    progress_frame.pack(fill=tk.X, pady=10, padx=5)
    
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, 
                                length=700, mode='determinate')
    progress_bar.pack(pady=10, fill=tk.X)
    
    status_label = ttk.Label(progress_frame, textvariable=status_var)
    status_label.pack(pady=5)
    
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