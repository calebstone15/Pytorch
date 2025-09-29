import os
import yaml

# Path to your dataset.yaml
yaml_path = "C:\\Users\\caleb\\OneDrive - Embry-Riddle Aeronautical University\\Documents\\Coding\\Pytorch\\Pytorch\\Data\\frankdatayolov1.1\\dataset.yaml"

# Load YAML file
with open(yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)
    
# Get base path from YAML
base_path = cfg['path']
train_path = os.path.join(base_path, cfg['train'])
labels_path = os.path.join(base_path, 'labels')

# Check paths
print(f"Checking if base path exists: {os.path.exists(base_path)}")
print(f"Checking if images path exists: {os.path.exists(train_path)}")
print(f"Checking if labels path exists: {os.path.exists(labels_path)}")

# Count files
if os.path.exists(train_path):
    image_files = os.listdir(train_path)
    print(f"Found {len(image_files)} files in images folder")
    
if os.path.exists(labels_path):
    label_files = os.listdir(labels_path)
    print(f"Found {len(label_files)} files in labels folder")
    
    # Check if some sample labels have content
    if label_files:
        sample = min(5, len(label_files))
        print(f"\nChecking content of {sample} sample label files:")
        for i, label_file in enumerate(label_files[:sample]):
            path = os.path.join(labels_path, label_file)
            size = os.path.getsize(path)
            print(f"  {label_file}: {size} bytes")
            
            # Read content of first few
            if size > 0:
                with open(path, 'r') as f:
                    content = f.read().strip()
                    print(f"    Content: {content[:50]}...")