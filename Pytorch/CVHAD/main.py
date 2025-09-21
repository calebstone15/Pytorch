import tkinter as tk
from LiveVideo import CameraHandler
from AnalyzeVideo import VideoAnalyzer

class CVHADApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("600x400")
        
        # Initialize handlers
        self.camera_handler = CameraHandler(self)
        self.video_analyzer = VideoAnalyzer(self)
        
        # Create GUI elements
        self.create_widgets()
        
        # Protocol for closing the window
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # Center content
        main_frame = tk.Frame(self.window)
        main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Application title
        self.title_label = tk.Label(main_frame, text="CVHAD", 
                                    font=("Arial", 24, "bold"))
        self.title_label.pack(pady=20)
        
        # Live Video button
        self.live_video_btn = tk.Button(main_frame, text="Live Video", 
                                        command=self.open_live_video_window, 
                                        width=25, height=3, 
                                        font=("Arial", 12))
        self.live_video_btn.pack(pady=15)
        
        # Analyze Video button
        self.analyze_video_btn = tk.Button(main_frame, text="Analyze Video", 
                                           command=self.open_analyze_video_window, 
                                           width=25, height=3,
                                           font=("Arial", 12))
        self.analyze_video_btn.pack(pady=15)
    
    def open_live_video_window(self):
        """Open a new window for live video functionality"""
        live_video_window = tk.Toplevel(self.window)
        live_video_window.title("Live Video")
        live_video_window.geometry("800x600")
        
        # Create live video interface
        self.camera_handler.create_interface(live_video_window)
    
    def open_analyze_video_window(self):
        """Open a new window for video analysis functionality"""
        analyze_video_window = tk.Toplevel(self.window)
        analyze_video_window.title("Analyze Video")
        analyze_video_window.geometry("800x600")
        
        # Create video analysis interface
        self.video_analyzer.create_interface(analyze_video_window)
    
    def on_closing(self):
        # Ensure any open windows are closed
        if hasattr(self.camera_handler, 'window') and self.camera_handler.window:
            self.camera_handler.close_window()
            
        if hasattr(self.video_analyzer, 'window') and self.video_analyzer.window:
            self.video_analyzer.close_window()
            
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CVHADApp(root, "CVHAD")
    root.mainloop()
