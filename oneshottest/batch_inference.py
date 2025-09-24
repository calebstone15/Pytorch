"""
Batch Inference Script for Burn Detection
Process multiple images in a directory with parallel processing for GPU acceleration
"""
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import sys
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

# Add the directory containing oneshotattempt2.py to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import necessary functions from your training script
from oneshotattempt2 import get_model, CLASSES, NUM_CLASSES

# Config file to store last used paths
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "batch_inference_config.json")

def load_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_model(model_path, device):
    """Load a trained model from disk"""
    try:
        print(f"Loading model from {model_path}...")
        model = get_model(NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image_batch(model, image_batch, device, confidence_threshold=0.5):
    """Process a batch of images in one forward pass"""
    with torch.no_grad():
        # Stack images into a batch tensor and move to device
        batch_tensor = torch.stack(image_batch).to(device)
        
        # Run inference on the batch
        predictions = model(batch_tensor)
    
    results = []
    for prediction in predictions:
        # Filter predictions by confidence threshold
        mask = prediction['scores'] >= confidence_threshold
        boxes = prediction['boxes'][mask].cpu().numpy()
        labels = prediction['labels'][mask].cpu().numpy()
        scores = prediction['scores'][mask].cpu().numpy()
        results.append((boxes, labels, scores))
    
    return results

def load_and_transform_image(image_path, transform):
    """Load and preprocess an image for inference"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        return image, image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

def save_detection_image(image, boxes, labels, scores, output_path):
    """Save an image with detection visualizations"""
    try:
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(np.array(image))
        
        # Draw boxes
        for box, label_idx, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor='b', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with score
            label_text = f"{CLASSES[label_idx]}: {score:.2f}"
            ax.text(
                x1, y1-5, label_text, 
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=12, color='white'
            )
        
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Error saving detection image {output_path}: {e}")
        return False

def process_directory(model, image_dir, output_dir, transform, device, confidence_threshold=0.5, batch_size=4):
    """Process all images in a directory with batching for GPU acceleration"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.isfile(os.path.join(image_dir, f)) and 
                  os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Determine optimal number of worker threads for loading images
    num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers maximum
    print(f"Using {num_workers} worker threads for image loading")
    
    # Process images in batches
    results = []
    
    # Calculate optimal batch size based on GPU memory
    if device.type == 'cuda':
        # Get GPU memory info
        if hasattr(torch.cuda, 'get_device_properties'):
            prop = torch.cuda.get_device_properties(device)
            total_memory = prop.total_memory / 1024**2  # Convert to MB
            # Adjust batch size based on available memory (heuristic)
            batch_size = min(max(int(total_memory / 1000), 1), 16)  # 1 to 16 batch size
            print(f"GPU has {total_memory:.0f} MB memory, using batch size of {batch_size}")
        else:
            # Default batch size if can't determine GPU memory
            batch_size = 4
            print(f"Using default batch size of {batch_size} for GPU")
    else:
        # For CPU, use smaller batch size
        batch_size = 2
        print(f"Using CPU with batch size of {batch_size}")
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i+batch_size]
        batch_paths = [os.path.join(image_dir, f) for f in batch_files]
        
        # Load images in parallel using ThreadPoolExecutor
        images = []
        tensors = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            load_results = list(executor.map(
                lambda path: load_and_transform_image(path, transform), 
                batch_paths
            ))
            
            for img, tensor in load_results:
                if img is not None and tensor is not None:
                    images.append(img)
                    tensors.append(tensor)
        
        if not tensors:
            continue  # Skip if no valid images in batch
            
        # Process batch on GPU
        batch_results = process_image_batch(model, tensors, device, confidence_threshold)
        
        # Process and save results in parallel
        batch_outputs = []
        for idx, (image_file, image, (boxes, labels, scores)) in enumerate(
            zip(batch_files[:len(images)], images, batch_results)
        ):
            base_name = os.path.splitext(image_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}_detected.png")
            
            # Save the detection image
            success = save_detection_image(image, boxes, labels, scores, output_path)
            
            if success:
                batch_outputs.append({
                    'image': image_file,
                    'detections': len(boxes),
                    'classes': [CLASSES[idx] for idx in labels],
                    'output': output_path
                })
        
        results.extend(batch_outputs)
    
    return results

def select_file_or_directory(title, is_file=False, initialdir=None):
    """Open a dialog to select a file or directory"""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    if is_file:
        path = filedialog.askopenfilename(
            title=title,
            initialdir=initialdir,
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
    else:
        path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()  # Properly destroy the tkinter window
    return path

def get_paths():
    """Get all necessary paths from user with a single tkinter instance"""
    config = load_config()
    
    # Create a single tkinter root window for all dialogs
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Select input directory
    print("Select input directory containing images...")
    input_dir = filedialog.askdirectory(
        title="Select Input Directory",
        initialdir=config.get('input_dir')
    )
    if not input_dir:
        root.destroy()
        return None, None, None
    
    # Select output directory
    print("Select output directory to save results...")
    output_dir = filedialog.askdirectory(
        title="Select Output Directory",
        initialdir=config.get('output_dir')
    )
    if not output_dir:
        root.destroy()
        return None, None, None
    
    # Always prompt to select model file
    print("Select model file (.pth)...")
    model_path = filedialog.askopenfilename(
        title="Select Model File (.pth)",
        initialdir=config.get('model_path_dir', os.path.dirname(__file__)),
        filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
    )
    if not model_path:
        root.destroy()
        return None, None, None
    
    root.destroy()
    
    # Save the config for next time
    config['input_dir'] = input_dir
    config['output_dir'] = output_dir
    config['model_path'] = model_path
    config['model_path_dir'] = os.path.dirname(model_path)
    save_config(config)
    
    return input_dir, output_dir, model_path

def main():
    # Get paths from user
    input_dir, output_dir, model_path = get_paths()
    
    if not input_dir or not output_dir or not model_path:
        print("Operation cancelled by user.")
        return
    
    # Validate paths
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Set up device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Set up image transformation
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Start timing
    start_time = time.time()
    
    # Process all images in the directory
    results = process_directory(
        model, 
        input_dir, 
        output_dir, 
        transform, 
        device, 
        confidence_threshold=0.5
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    if results:
        print("\nProcessing Summary:")
        print(f"Processed {len(results)} images in {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/len(results):.4f} seconds")
        print(f"Total detections: {sum(r['detections'] for r in results)}")
        print(f"Results saved to {output_dir}")
        
        # Count detections by class
        class_counts = {}
        for r in results:
            for cls in r['classes']:
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print("\nDetections by class:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()