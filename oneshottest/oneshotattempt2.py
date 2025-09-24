# Object Detection for Burn Analysis with PyTorch
# This script trains and evaluates a Faster R-CNN model for object detection
# It uses COCO-format annotations and supports GPU acceleration with CUDA

# Import necessary libraries
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# Define constants for class labels
CLASSES = ['background', 'nominal', 'off-nominal', 'fire', 'melting', 'fluid_leak', 'venting_smoke']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

# Dataset class for loading COCO-format annotations
class COCODataset(Dataset):
    """Dataset for COCO format annotations"""
    
    def __init__(self, annotations_file, images_dir, transform=None):
        """
        Args:
            annotations_file: Path to the COCO format JSON annotation file
            images_dir: Directory containing the images
            transform: Optional transform to be applied to images
        """
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.transform = transform
        
        # Parse annotations
        self.annotations = []
        self._parse_annotations()
    
    def _parse_annotations(self):
        """Parse COCO format JSON annotations file and build a list of image annotations"""
        # Check if annotations file exists
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        # Check if images directory exists
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
        # Load COCO annotations
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
            
        # Print structure of COCO data for debugging
        print(f"COCO data keys: {coco_data.keys()}")
        print(f"Number of images: {len(coco_data.get('images', []))}")
        print(f"Number of categories: {len(coco_data.get('categories', []))}")
        print(f"Number of annotations: {len(coco_data.get('annotations', []))}")
        
        if 'images' not in coco_data or not coco_data['images']:
            raise ValueError("No images found in the COCO annotation file")
            
        if 'categories' not in coco_data or not coco_data['categories']:
            raise ValueError("No categories found in the COCO annotation file")
            
        if 'annotations' not in coco_data or not coco_data['annotations']:
            raise ValueError("No annotations found in the COCO annotation file")
        
        # Print category information for debugging
        print("Categories in annotations:")
        for cat in coco_data['categories']:
            print(f"  ID: {cat['id']}, Name: {cat['name']}")
        
        # Build dictionaries for quick lookup
        images_dict = {img['id']: img for img in coco_data['images']}
        categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image_id
        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Process each image and its annotations
        for img_id, anns in img_to_anns.items():
            # Get image info
            img_info = images_dict[img_id]
            image_name = img_info['file_name']
            image_path = os.path.join(self.images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping")
                continue
            
            width = img_info['width']
            height = img_info['height']
            
            boxes = []
            labels = []
            
            # Process annotations for this image
            for ann in anns:
                # Get category name and convert to our class index
                cat_name = categories_dict[ann['category_id']]
                
                # Map COCO category name to our class names
                # Try direct matching first (exact match)
                if cat_name.lower() in CLASS_TO_IDX:
                    label = cat_name.lower()
                # Try to map based on substring matching
                elif 'nominal' in cat_name.lower() and 'off' not in cat_name.lower():
                    label = 'nominal'
                elif 'off-nominal' in cat_name.lower() or 'off_nominal' in cat_name.lower() or 'offnominal' in cat_name.lower():
                    label = 'off-nominal'
                elif 'fire' in cat_name.lower():
                    label = 'fire'
                elif 'melting' in cat_name.lower() or 'melt' in cat_name.lower():
                    label = 'melting'
                elif 'fluid' in cat_name.lower() or 'leak' in cat_name.lower():
                    label = 'fluid_leak'
                elif 'venting' in cat_name.lower() or 'smoke' in cat_name.lower() or 'vent' in cat_name.lower():
                    label = 'venting_smoke'
                else:
                    print(f"Warning: Unknown category '{cat_name}', skipping")
                    continue
                    
                print(f"Mapped '{cat_name}' to '{label}'")
                
                if label not in CLASS_TO_IDX:
                    print(f"Warning: Mapped label {label} not in CLASS_TO_IDX, skipping")
                    continue
                
                # Get bounding box
                bbox = ann['bbox']  # [x, y, width, height] in COCO format
                x_min, y_min, w, h = bbox
                x_max = x_min + w
                y_max = y_min + h
                
                # Normalize to [0, 1]
                boxes.append([x_min/width, y_min/height, x_max/width, y_max/height])
                labels.append(CLASS_TO_IDX[label])
            
            if boxes and labels:
                self.annotations.append({
                    'image_path': image_path,
                    'image_name': image_name,
                    'width': width,
                    'height': height,
                    'boxes': boxes,
                    'labels': labels
                })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        img = Image.open(ann['image_path']).convert("RGB")
        
        # Convert normalized boxes back to pixel coordinates
        width, height = ann['width'], ann['height']
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        boxes[:, 0] *= width  # x_min
        boxes[:, 1] *= height  # y_min
        boxes[:, 2] *= width  # x_max
        boxes[:, 3] *= height  # y_max
        
        # Create target dict for Faster R-CNN
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.tensor(ann['labels'], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, target

def get_model(num_classes):
    """
    Load a pre-trained Faster R-CNN model and modify it for our custom classes
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the classifier with a new one for our custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    """Custom collate function for DataLoader to handle variable sized images and targets"""
    return tuple(zip(*batch))

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    Each box is [x1, y1, x2, y2] format
    """
    # Get the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def evaluate_accuracy(model, data_loader, device, iou_threshold=0.5):
    """Evaluate the model on the validation set and return accuracy metrics"""
    model.eval()
    total_correct = 0
    total_objects = 0
    total_predictions = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, (target, output) in enumerate(zip(targets, outputs)):
                target_boxes = target['boxes'].cpu().numpy()
                target_labels = target['labels'].cpu().numpy()
                
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                
                # Filter predictions by confidence threshold
                confident_preds = pred_scores >= 0.5
                pred_boxes = pred_boxes[confident_preds]
                pred_labels = pred_labels[confident_preds]
                
                # Count total objects and predictions
                total_objects += len(target_boxes)
                total_predictions += len(pred_boxes)
                
                # For each ground truth box, check if there's a matching prediction
                for t_box, t_label in zip(target_boxes, target_labels):
                    found_match = False
                    for p_box, p_label in zip(pred_boxes, pred_labels):
                        # Check if labels match and IoU is above threshold
                        if p_label == t_label and calculate_iou(t_box, p_box) >= iou_threshold:
                            found_match = True
                            total_correct += 1
                            break
    
    # Calculate metrics
    precision = total_correct / max(total_predictions, 1)
    recall = total_correct / max(total_objects, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': total_correct / max(total_objects, 1)
    }

def train_one_epoch(model, optimizer, data_loader, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def plot_metrics(train_losses, val_metrics, save_path=None):
    """Plot training and validation metrics over epochs"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation metrics
    for metric_name in ['precision', 'recall', 'f1_score', 'accuracy']:
        if metric_name in val_metrics[0]:
            values = [metrics[metric_name] for metrics in val_metrics]
            ax2.plot(epochs, values, label=metric_name.capitalize())
    
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, filename):
    """Save the trained model"""
    torch.save(model.state_dict(), filename)

def load_model(filename, num_classes):
    """Load a trained model"""
    model = get_model(num_classes)
    model.load_state_dict(torch.load(filename))
    return model

def format_time(seconds):
    """Format seconds into a human-readable time string"""
    return str(timedelta(seconds=int(seconds)))

def visualize_predictions(model, dataset, device, num_images=5, confidence_threshold=0.5):
    """Visualize model predictions on sample images"""
    model.eval()
    
    # Ensure we don't try to visualize more images than exist in the dataset
    num_images = min(num_images, len(dataset))
    
    if num_images == 0:
        print("No images to visualize!")
        return
    
    # Create figure with appropriate size based on number of images
    fig, axs = plt.subplots(num_images, 2, figsize=(15, 5*num_images))
    
    # Handle the case where there's only one image (axs would be 1D)
    if num_images == 1:
        axs = axs.reshape(1, -1)
    
    for i in range(num_images):
        try:
            # Get image and target
            img, target = dataset[i]
            
            # Convert image tensor for display
            # If image is already a tensor with shape [C,H,W]
            if isinstance(img, torch.Tensor) and img.dim() == 3:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                # Normalize to [0,1] range if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                # Clip to valid range
                img_np = np.clip(img_np, 0, 1)
            else:
                # If image is not a tensor or has unexpected shape
                print(f"Warning: Unexpected image format for image {i}, skipping")
                continue
            
            # Show ground truth
            axs[i, 0].imshow(img_np)
            axs[i, 0].set_title('Ground Truth')
            axs[i, 0].axis('off')
            
            # Draw ground truth boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axs[i, 0].add_patch(rect)
                    
                    # Use label to get class name, ensure it's a valid index
                    class_name = CLASSES[label] if 0 <= label < len(CLASSES) else f"Unknown ({label})"
                    axs[i, 0].text(
                        x1, y1, class_name, 
                        bbox=dict(facecolor='yellow', alpha=0.5),
                        fontsize=10, color='black'
                    )
            else:
                axs[i, 0].text(10, 10, "No annotations", fontsize=12)
            
            # Make prediction
            with torch.no_grad():
                # Ensure image is on the right device
                prediction = model([img.to(device)])[0]
            
            # Show prediction
            axs[i, 1].imshow(img_np)
            axs[i, 1].set_title('Prediction')
            axs[i, 1].axis('off')
            
            # Check if we have any predictions
            if len(prediction['boxes']) > 0:
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                
                # Filter by confidence
                mask = scores >= confidence_threshold
                filtered_boxes = boxes[mask]
                filtered_labels = labels[mask]
                filtered_scores = scores[mask]
                
                # Display predictions
                for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='b', facecolor='none'
                    )
                    axs[i, 1].add_patch(rect)
                    
                    # Use label to get class name, ensure it's a valid index
                    class_name = CLASSES[label] if 0 <= label < len(CLASSES) else f"Unknown ({label})"
                    axs[i, 1].text(
                        x1, y1, f"{class_name}: {score:.2f}", 
                        bbox=dict(facecolor='cyan', alpha=0.5),
                        fontsize=10, color='black'
                    )
                
                if len(filtered_boxes) == 0:
                    axs[i, 1].text(10, 10, "No detections above threshold", fontsize=12)
            else:
                axs[i, 1].text(10, 10, "No detections", fontsize=12)
                
        except Exception as e:
            print(f"Error visualizing image {i}: {e}")
            # If there's an error, create an empty plot with error message
            if i < num_images:
                axs[i, 0].text(0.5, 0.5, f"Error: {str(e)}", 
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=axs[i, 0].transAxes)
                axs[i, 1].text(0.5, 0.5, f"Error: {str(e)}", 
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=axs[i, 1].transAxes)
                axs[i, 0].axis('off')
                axs[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("prediction_visualization.png")
    plt.show()

def main():
    # Check for GPU availability with graceful fallback
    print(f"PyTorch version: {torch.__version__}")
    print("Checking PyTorch CUDA configuration:")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Determine device with fallback to CPU if CUDA fails
    try:
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # Try to use CUDA
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Print CUDA memory information
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            batch_size = 2  # Can use larger batch size with GPU
            num_workers = 2  # Can use workers with GPU
        else:
            device = torch.device('cpu')
            batch_size = 1  # Smaller batch size for CPU
            num_workers = 0  # No workers for CPU
            print("CUDA not available, using CPU instead.")
            print("For training on GPU with Python 3.11, install PyTorch with CUDA support:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        device = torch.device('cpu')
        batch_size = 1
        num_workers = 0
        print(f"Error checking CUDA availability: {e}")
        print("Falling back to CPU.")
    
    print(f"Using device: {device}")
    
    # Data directories - replace with your actual paths
    annotations_file = "C:\\Users\\caleb\\OneDrive - Embry-Riddle Aeronautical University\\Documents\\Coding\\Schiliren\\Pytorch\\Data\\oneshottest\\annotations\\instances_default.json"
    images_dir = "C:\\Users\\caleb\\OneDrive - Embry-Riddle Aeronautical University\\Documents\\Coding\\Schiliren\\Pytorch\\Data\\oneshottest\\images"

    # Check if files exist
    print(f"Checking if annotation file exists: {os.path.exists(annotations_file)}")
    print(f"Checking if images directory exists: {os.path.exists(images_dir)}")
    
    if not os.path.exists(annotations_file):
        print(f"ERROR: Annotation file not found at: {annotations_file}")
        return
        
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found at: {images_dir}")
        return

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataset
    try:
        dataset = COCODataset(annotations_file, images_dir, transform=transform)
        print(f"Dataset contains {len(dataset)} images")
        
        if len(dataset) == 0:
            print("ERROR: Dataset is empty. Please check your annotations and image files.")
            return
            
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        if train_size == 0 or val_size == 0:
            print("ERROR: Not enough data to split into train and validation sets.")
            return
            
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        print(f"Training set: {len(train_dataset)} images")
        print(f"Validation set: {len(val_dataset)} images")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    # Create model
    model = get_model(NUM_CLASSES)
    model.to(device)
    
    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Train model
    num_epochs = 50  # Changed from conditional 10/5 to fixed 50 epochs
    
    # Metrics tracking
    train_losses = []
    val_metrics = []
    total_start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()
            loss = train_one_epoch(model, optimizer, train_loader, device)
            train_losses.append(loss)
            lr_scheduler.step()
            
            # Evaluate on validation set
            print("Evaluating on validation set:")
            metrics = evaluate_accuracy(model, val_loader, device)
            val_metrics.append(metrics)
            
            end_time = time.time()
            epoch_time = end_time - start_time
            elapsed_time = end_time - total_start_time
            estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
            remaining_time = estimated_total - elapsed_time
            
            print(f"  Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            print(f"  Epoch time: {format_time(epoch_time)}, Elapsed: {format_time(elapsed_time)}, Remaining: {format_time(remaining_time)}")
            
            # Save checkpoint after each epoch
            checkpoint_path = f"burn_detection_checkpoint_epoch_{epoch+1}.pth"
            save_model(model, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
            
            # Removed plotting after each epoch to avoid interrupting training
    
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model...")
        save_model(model, "burn_detection_interrupted.pth")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate and display total training time
    total_training_time = time.time() - total_start_time
    print(f"Total training time: {format_time(total_training_time)}")
    
    # Save the final trained model
    save_model(model, "burn_detection_model_final.pth")
    
    # Plot final metrics only at the end of training
    print("Generating final training metrics plot...")
    plot_metrics(train_losses, val_metrics, "final_training_metrics.png")
    
    # Visualize some predictions
    print("Generating prediction visualizations...")
    visualize_predictions(model, val_dataset, device)

if __name__ == "__main__":
    main()