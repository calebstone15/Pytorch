import os
import shutil
import tkinter as tk
from collections import defaultdict
from tkinter import filedialog, messagebox

import defusedxml.ElementTree as ET

def parse_points(points_str):
    """Parse CVAT polygon points which might be in format 'x1,y1;x2,y2;...' or 'x1 y1 x2 y2 ...'"""
    if ';' in points_str:
        # Format is 'x1,y1;x2,y2;...'
        points = []
        for point_pair in points_str.split(';'):
            if ',' in point_pair:
                x_coord, y_coord = point_pair.split(',')
                points.extend([float(x_coord), float(y_coord)])
        return points

    # Format is 'x1 y1 x2 y2 ...'
    return [float(p) for p in points_str.split()]

def _setup_directories(base_dir):
    """Setup output directories for labels and images."""
    output_dir = os.path.join(base_dir, 'yolo_output')
    # Use a relative path or check if the specific user path exists
    # otherwise default to 'images' in base_dir
    user_images_path = os.path.join(base_dir,
                                   "C:\\Users\\caleb\\Downloads\\cvatcvat3\\images\\train")
    images_dir = user_images_path if os.path.exists(user_images_path) \
                 else os.path.join(base_dir, 'images')
    if not os.path.exists(images_dir):
        images_dir = None

    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images') if images_dir else None

    os.makedirs(labels_dir, exist_ok=True)
    if images_output_dir:
        os.makedirs(images_output_dir, exist_ok=True)

    return output_dir, labels_dir, images_dir, images_output_dir

def _get_default_class_mapping():
    """Returns the predefined class mapping."""
    # Use predefined class mapping
    mapping = {
        "Off-Nominal": 0,
        "Nominal": 1,
        "Fire": 2, 
        "Melting": 3,
        "Fluid Leak": 4,
        "Venting/Smoke": 5
    }
    print("Using predefined class mapping:", mapping)
    return mapping

def _extract_box_data(elem):
    """Extracts bounding box data from a box or polygon element."""
    if elem.tag == 'box':
        xtl = float(elem.get('xtl'))
        ytl = float(elem.get('ytl'))
        xbr = float(elem.get('xbr'))
        ybr = float(elem.get('ybr'))
    else:  # polygon: compute bbox from points
        points_str = elem.get('points')
        points = parse_points(points_str)
        xs_points = points[0::2]
        ys_points = points[1::2]
        xtl, ytl = min(xs_points), min(ys_points)
        xbr, ybr = max(xs_points), max(ys_points)
    return xtl, ytl, xbr, ybr

def _process_video_track(xml_root, mapping):
    """Processes video tracks and groups annotations by frame."""
    annotations_by_frame = defaultdict(list)
    for track in xml_root.findall('track'):
        label = track.get('label')
        if label not in mapping:
            continue
        class_id = mapping[label]
        for elem in track.findall('box') + track.findall('polygon'):
            frame = int(elem.get('frame'))
            xtl, ytl, xbr, ybr = _extract_box_data(elem)
            annotations_by_frame[frame].append((class_id, xtl, ytl, xbr, ybr))
    return annotations_by_frame

def _write_label_file(txt_path, annotations, img_dims):
    """Writes annotations to a text file in YOLO format."""
    img_width, img_height = img_dims
    with open(txt_path, 'w', encoding='utf-8') as f:
        for class_id, xtl, ytl, xbr, ybr in annotations:
            width = (xbr - xtl) / img_width
            height = (ybr - ytl) / img_height
            x_center = (xtl + (xbr - xtl) / 2) / img_width
            y_center = (ytl + (ybr - ytl) / 2) / img_height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}\n")

def _copy_image(img_name, src_dir, dst_dir):
    """Copies image from source to output directory."""
    if src_dir and dst_dir:
        src_img = os.path.join(src_dir, img_name)
        dst_img = os.path.join(dst_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"Warning: Image {img_name} not found in {src_dir}. Skipping copy.")

def _process_video_task(xml_root, mapping, labels_dir, images_dir, images_output_dir):
    """Handles processing for video tasks."""
    annotations_by_frame = _process_video_track(xml_root, mapping)

    # Get video size
    original_size = xml_root.find(".//original_size")
    if original_size is None:
        print("Warning: No original_size found. Assuming 1920x1080.")
        img_width, img_height = 1920, 1080
    else:
        img_width = float(original_size.find('width').text)
        img_height = float(original_size.find('height').text)

    for frame, anns in annotations_by_frame.items():
        img_name = f"frame_{frame:06d}.jpg"
        txt_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        _write_label_file(txt_path, anns, (img_width, img_height))
        _copy_image(img_name, images_dir, images_output_dir)

def _process_image_task(xml_root, mapping, labels_dir, images_dir, images_output_dir):
    """Handles processing for image tasks."""
    for image in xml_root.findall('image'):
        img_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))

        annotations = []

        # Handle boxes and polygons
        for elem in image.findall('box') + image.findall('polygon'):
            label = elem.get('label')
            if label not in mapping:
                continue
            class_id = mapping[label]
            xtl, ytl, xbr, ybr = _extract_box_data(elem)
            annotations.append((class_id, xtl, ytl, xbr, ybr))

        txt_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        _write_label_file(txt_path, annotations, (img_width, img_height))
        _copy_image(img_name, images_dir, images_output_dir)

def _create_data_yaml(output_dir, mapping):
    """Creates the data.yaml file."""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("path: .\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(mapping)}\n")
        f.write("names:\n")
        for id_, label in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"  {id_}: {label}\n")

def convert_cvat_xml_to_yolo(path_to_xml, mapping=None):
    """
    Convert CVAT XML annotations to YOLO format, handling boxes, polygons, and video tracks.
    Debug prints structure if no annotations found.

    Args:
        path_to_xml (str): Path to the CVAT XML file.
        mapping (dict, optional): Dictionary mapping label names to class IDs. If None, auto-assign.

    Returns:
        None
    """
    base_dir = os.path.dirname(path_to_xml)
    output_dir, labels_dir, images_dir, images_output_dir = _setup_directories(base_dir)

    tree = ET.parse(path_to_xml)
    xml_root = tree.getroot()

    # Debug: Print XML structure
    print("XML root tag:", xml_root.tag)
    print("Number of images:", len(xml_root.findall('image')))
    print("Number of tracks:", len(xml_root.findall('track')))

    # Determine if video task
    task_mode = xml_root.find(".//task/mode")
    is_video = len(xml_root.findall('track')) > 0 or \
               (task_mode is not None and task_mode.text == 'interpolation')
    print("Detected video task?", is_video)

    # Use predefined class mapping if not provided
    if mapping is None:
        mapping = _get_default_class_mapping()

    if is_video:
        _process_video_task(xml_root, mapping, labels_dir, images_dir, images_output_dir)
    else:
        _process_image_task(xml_root, mapping, labels_dir, images_dir, images_output_dir)

    _create_data_yaml(output_dir, mapping)

    print(f"Conversion complete. Output saved to {output_dir}.")
    if images_dir is None:
        print("No 'images' folder found. Labels generated without images.")
    print(f"Generated {len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])} label files.")

if __name__ == "__main__":
    tk_root = tk.Tk()
    tk_root.withdraw()

    file_path = filedialog.askopenfilename(title="Select CVAT XML", filetypes=[("XML", "*.xml")])
    if not file_path:
        exit()

    CLASS_MAPPING = None  # Auto

    convert_cvat_xml_to_yolo(file_path, CLASS_MAPPING)
