import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
from collections import defaultdict

def parse_points(points_str):
    """Parse CVAT polygon points which might be in format 'x1,y1;x2,y2;...' or 'x1 y1 x2 y2 ...'"""
    if ';' in points_str:
        # Format is 'x1,y1;x2,y2;...'
        points = []
        for point_pair in points_str.split(';'):
            if ',' in point_pair:
                x, y = point_pair.split(',')
                points.extend([float(x), float(y)])
        return points
    else:
        # Format is 'x1 y1 x2 y2 ...'
        return [float(p) for p in points_str.split()]

def convert_cvat_xml_to_yolo(xml_path, class_mapping=None):
    """
    Convert CVAT XML annotations to YOLO format, handling boxes, polygons, and video tracks.
    Debug prints structure if no annotations found.

    Args:
        xml_path (str): Path to the CVAT XML file.
        class_mapping (dict, optional): Dictionary mapping label names to class IDs. If None, auto-assign.

    Returns:
        None
    """
    base_dir = os.path.dirname(xml_path)
    output_dir = os.path.join(base_dir, 'yolo_output')
    images_dir = os.path.join(base_dir, "C:\\Users\\caleb\\Downloads\\cvatcvat3\\images\\train"
                              ) if os.path.exists(os.path.join(base_dir, 'images')) else None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Debug: Print XML structure
    print("XML root tag:", root.tag)
    print("Number of images:", len(root.findall('image')))
    print("Number of tracks:", len(root.findall('track')))

    # Determine if video task
    task_mode = root.find(".//task/mode")
    is_video = len(root.findall('track')) > 0 or (task_mode is not None and task_mode.text == 'interpolation')
    print("Detected video task?", is_video)

    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images') if images_dir else None
    os.makedirs(labels_dir, exist_ok=True)
    if images_output_dir:
        os.makedirs(images_output_dir, exist_ok=True)

    # Auto-create class mapping
    labels = set()
    if is_video:
        for track in root.findall('track'):
            labels.add(track.get('label'))
        # Also check box/poly in tracks
        for track in root.findall('track'):
            for box in track.findall('box'):
                labels.add(box.get('label'))
            for polygon in track.findall('polygon'):
                labels.add(polygon.get('label'))
    else:
        for image in root.findall('image'):
            for box in image.findall('box'):
                labels.add(box.get('label'))
            for polygon in image.findall('polygon'):
                labels.add(polygon.get('label'))

    if not labels:
        messagebox.showerror("No Annotations", "No labels or annotations found in XML. Check if boxes/polygons exist.")
        return

    # Replace the auto-generated class mapping with this fixed mapping
    # class_mapping = {label: i for i, label in enumerate(sorted(labels))}
    # print("Auto-generated class mapping:", class_mapping)

    # Use your predefined class mapping instead
    class_mapping = {
        "Off-Nominal": 0,
        "Nominal": 1,
        "Fire": 2, 
        "Melting": 3,
        "Fluid Leak": 4,
        "Venting/Smoke": 5
    }
    print("Using predefined class mapping:", class_mapping)

    if is_video:
        # Video task: group by frame
        annotations_by_frame = defaultdict(list)
        for track in root.findall('track'):
            label = track.get('label')
            if label not in class_mapping:
                continue
            class_id = class_mapping[label]
            for elem in track.findall('box') + track.findall('polygon'):
                frame = int(elem.get('frame'))
                if elem.tag == 'box':
                    xtl = float(elem.get('xtl'))
                    ytl = float(elem.get('ytl'))
                    xbr = float(elem.get('xbr'))
                    ybr = float(elem.get('ybr'))
                else:  # polygon: compute bbox from points
                    points_str = elem.get('points')
                    points = parse_points(points_str)
                    xs = points[0::2]
                    ys = points[1::2]
                    xtl, ytl = min(xs), min(ys)
                    xbr, ybr = max(xs), max(ys)
                annotations_by_frame[frame].append((class_id, xtl, ytl, xbr, ybr))

        # Get video size
        original_size = root.find(".//original_size")
        if original_size is None:
            print("Warning: No original_size found. Assuming 1920x1080.")
            img_width, img_height = 1920, 1080
        else:
            img_width = float(original_size.find('width').text)
            img_height = float(original_size.find('height').text)

        # Start frame
        start_frame_elem = root.find(".//start_frame")
        start_frame = int(start_frame_elem.text) if start_frame_elem is not None else 0

        for frame, anns in annotations_by_frame.items():
            img_name = f"frame_{frame:06d}.jpg"  # Adjust padding if needed
            txt_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

            with open(txt_path, 'w') as f:
                for class_id, xtl, ytl, xbr, ybr in anns:
                    width = (xbr - xtl) / img_width
                    height = (ybr - ytl) / img_height
                    x_center = (xtl + (xbr - xtl) / 2) / img_width
                    y_center = (ytl + (ybr - ytl) / 2) / img_height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if images_dir and images_output_dir:
                src_img = os.path.join(images_dir, img_name)
                dst_img = os.path.join(images_output_dir, img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                else:
                    print(f"Warning: Image {img_name} not found in {images_dir}. Skipping copy.")

    else:
        # Image task
        for image in root.findall('image'):
            img_name = image.get('name')
            img_width = float(image.get('width'))
            img_height = float(image.get('height'))

            txt_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

            with open(txt_path, 'w') as f:
                # Handle boxes
                for box in image.findall('box'):
                    label = box.get('label')
                    if label not in class_mapping:
                        continue
                    class_id = class_mapping[label]
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))
                    width = (xbr - xtl) / img_width
                    height = (ybr - ytl) / img_height
                    x_center = (xtl + (xbr - xtl) / 2) / img_width
                    y_center = (ytl + (ybr - ytl) / 2) / img_height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Handle polygons (convert to bbox)
                for polygon in image.findall('polygon'):
                    label = polygon.get('label')
                    if label not in class_mapping:
                        continue
                    class_id = class_mapping[label]
                    points_str = polygon.get('points')
                    points = parse_points(points_str)
                    xs = points[0::2]
                    ys = points[1::2]
                    xtl, ytl = min(xs), min(ys)
                    xbr, ybr = max(xs), max(ys)
                    width = (xbr - xtl) / img_width
                    height = (ybr - ytl) / img_height
                    x_center = (xtl + (xbr - xtl) / 2) / img_width
                    y_center = (ytl + (ybr - ytl) / 2) / img_height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            if images_dir and images_output_dir:
                src_img = os.path.join(images_dir, img_name)
                dst_img = os.path.join(images_output_dir, img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                else:
                    print(f"Warning: Image {img_name} not found in {images_dir}. Skipping copy.")

    # Create data.yaml
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write("path: .\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(class_mapping)}\n")
        f.write("names:\n")
        for id_, label in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"  {id_}: {label}\n")

    print(f"Conversion complete. Output saved to {output_dir}.")
    if images_dir is None:
        print("No 'images' folder found. Labels generated without images.")
    print(f"Generated {len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])} label files.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    xml_path = filedialog.askopenfilename(title="Select CVAT XML", filetypes=[("XML", "*.xml")])
    if not xml_path:
        exit()

    class_mapping = None  # Auto

    convert_cvat_xml_to_yolo(xml_path, class_mapping)