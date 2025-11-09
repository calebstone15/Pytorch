import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import ttk
import subprocess
import os

class ConverterApp:
    def __init__(self, root):
        self.root = root
        root.title("MP4 Re-Wrapper")
        root.geometry("600x400")
        
        self.file_paths = []
        self.output_pattern_var = tk.StringVar(value="{name}_playable{ext}")

        # --- Frame for Title ---
        title_frame = tk.Frame(root, pady=10)
        title_frame.pack()
        
        self.title_label = tk.Label(title_frame, text="Pi Camera Recording Fixer", font=("Helvetica", 16, "bold"), fg="#1e293b")
        self.title_label.pack()
        
        self.info_label = tk.Label(title_frame, text="Converts unplayable H.264 files into proper MP4s.", font=("Helvetica", 10), fg="#4b5563")
        self.info_label.pack()

        # --- Frame for Buttons ---
        button_frame = tk.Frame(root, pady=10)
        button_frame.pack()

        self.select_button = tk.Button(button_frame, text="1. Select File(s)", command=self.select_files, font=("Helvetica", 12), width=20)
        self.select_button.pack(side=tk.LEFT, padx=10)

        self.convert_button = tk.Button(button_frame, text="2. Convert", command=self.convert_files, font=("Helvetica", 12, "bold"), width=20, state=tk.DISABLED)
        self.convert_button.pack(side=tk.LEFT, padx=10)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=30, pady=5)

        # --- Output Options ---
        options_frame = tk.Frame(root, padx=20, pady=5)
        options_frame.pack(fill=tk.X)
        options_frame.grid_columnconfigure(0, weight=1)

        tk.Label(options_frame, text="Output filename pattern", font=("Helvetica", 11, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(options_frame, text="Use {name} for the original filename and {ext} for the extension.", font=("Helvetica", 9), fg="#4b5563", anchor="w").grid(row=1, column=0, sticky="w", pady=(2, 6))
        self.pattern_entry = tk.Entry(options_frame, textvariable=self.output_pattern_var, font=("Helvetica", 11), width=35)
        self.pattern_entry.grid(row=2, column=0, sticky="we")

        # --- Frame for Status Log ---
        log_frame = tk.Frame(root, padx=20, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_label = tk.Label(log_frame, text="Status:", font=("Helvetica", 10, "italic"), anchor="w")
        self.log_label.pack(fill="x")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, font=("Menlo", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def log(self, message):
        """Adds a message to the log text box."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Auto-scroll to the bottom
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def check_ffmpeg(self):
        """Checks if ffmpeg is installed and accessible."""
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("ERROR: ffmpeg not found!")
            self.log("Please install ffmpeg on your Mac.")
            self.log("The easiest way is to open Terminal and run:")
            self.log("brew install ffmpeg")
            self.convert_button.config(state=tk.DISABLED)
            return False

    def select_files(self):
        """Opens a file dialog to select .mp4 files."""
        # This will open the native macOS file dialog
        self.file_paths = filedialog.askopenfilenames(
            title="Select unplayable .mp4 files",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if self.file_paths:
            self.log(f"Selected {len(self.file_paths)} file(s):")
            for f in self.file_paths:
                self.log(f"  - {os.path.basename(f)}")
            # Only enable convert button if ffmpeg is found
            if self.check_ffmpeg():
                self.convert_button.config(state=tk.NORMAL)
        else:
            self.log("No files selected.")
            self.convert_button.config(state=tk.DISABLED)

    def convert_files(self):
        """Runs the ffmpeg conversion process."""
        if not self.file_paths:
            self.log("No files to convert. Please select files first.")
            return

        pattern = self.output_pattern_var.get().strip() or "{name}_playable{ext}"

        self.log("\nStarting conversion...")
        self.log(f"Using pattern: {pattern}")
        self.select_button.config(state=tk.DISABLED)
        self.convert_button.config(state=tk.DISABLED)
        
        success_count = 0
        for in_path in self.file_paths:
            try:
                dir_name = os.path.dirname(in_path)
                base_name = os.path.basename(in_path)
                file_name, file_ext = os.path.splitext(base_name)
                file_ext = file_ext or ".mp4"

                final_name = pattern.replace("{name}", file_name).replace("{ext}", file_ext)
                if not os.path.splitext(final_name)[1]:
                    final_name += file_ext

                if os.path.abspath(os.path.join(dir_name, final_name)) == os.path.abspath(in_path):
                    final_name = f"{file_name}_converted{file_ext}"

                final_ext = os.path.splitext(final_name)[1]
                name_root = final_name[:-len(final_ext)] if final_ext else final_name
                candidate_name = final_name
                counter = 1
                while os.path.exists(os.path.join(dir_name, candidate_name)):
                    candidate_name = f"{name_root}_{counter}{final_ext}"
                    counter += 1

                out_path = os.path.join(dir_name, candidate_name)

                self.log(f"Converting {base_name}...")
                self.log(f"  -> Saving as {os.path.basename(out_path)}")
                
                # The magic ffmpeg command to "re-wrap" the video
                command = [
                    "ffmpeg",
                    "-i", in_path,     # Input file
                    "-c", "copy",      # Codec: copy (no re-encoding)
                    "-y",              # Overwrite output file without asking
                    out_path           # Output file
                ]
                
                # Run the command
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.log(f"SUCCESS: Saved {os.path.basename(out_path)}")
                success_count += 1
                
            except subprocess.CalledProcessError as e:
                self.log(f"ERROR converting {base_name}:")
                self.log(f"  {e.stderr.decode('utf-8')}")
            except Exception as e:
                self.log(f"An unexpected error occurred: {e}")

        self.log(f"\n--- Conversion Finished ---")
        self.log(f"Successfully converted {success_count} / {len(self.file_paths)} files.")
        
        # Reset
        self.file_paths = []
        self.select_button.config(state=tk.NORMAL)
        self.convert_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    # Check for ffmpeg on app start
    root.after(100, app.check_ffmpeg)
    root.mainloop()