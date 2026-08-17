import os
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image

# ============================================================
# Root directory of the YOLO dataset (includes images, labels and classes.txt)
YOLO_ROOT = Path("/LGPSD-DET")
# Output directory for the converted VOC dataset
VOC_OUTPUT_ROOT = Path("/LGPSD-DET_VOC")
# Path to the class definition file
CLASSES_FILE = YOLO_ROOT / "classes.txt"
# Image file extension in your dataset
IMAGE_EXT = ".png"
# Dataset splits to process
SPLITS = ["train", "val", "test"]

# Automatically load class names from classes.txt (index order matches YOLO labels)
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]


def create_voc_xml(xml_save_path: Path, img_filename: str, img_size: tuple, objects: list):
    """
    Generate a standard PASCAL VOC XML annotation file.

    Args:
        xml_save_path: Full file path to save the output XML
        img_filename: Original image file name
        img_size: Tuple of (image_width, image_height, channel_depth)
        objects: List of defect objects, each contains 'name' and 'bbox' (xmin, ymin, xmax, ymax)
    """
    # Create root annotation node
    annotation = ET.Element("annotation")

    # Basic file metadata
    folder = ET.SubElement(annotation, "folder")
    folder.text = "JPEGImages"
    filename = ET.SubElement(annotation, "filename")
    filename.text = img_filename

    # Dataset source information
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "LGPSD-DET Dataset"

    # Image dimension information
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_size[0])
    height = ET.SubElement(size, "height")
    height.text = str(img_size[1])
    depth = ET.SubElement(size, "depth")
    depth.text = str(img_size[2])

    # Segmentation flag (default 0 for object detection tasks)
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Write each defect object into XML
    for obj in objects:
        obj_node = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj_node, "name")
        name.text = obj["name"]
        pose = ET.SubElement(obj_node, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj_node, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj_node, "difficult")
        difficult.text = "0"

        # Bounding box coordinates
        bndbox = ET.SubElement(obj_node, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(round(obj["bbox"][0])))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(round(obj["bbox"][1])))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(round(obj["bbox"][2])))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(round(obj["bbox"][3])))

    # Format and write XML file
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="\t", level=0)
    tree.write(xml_save_path, encoding="utf-8", xml_declaration=True)


def yolo_to_voc():
    """
    Main conversion function.
    Output standard VOC directory structure:
    LGPSD-DET_VOC/
    ├── Annotations/      # All XML annotation files
    ├── JPEGImages/       # All raw .png images
    └── ImageSets/
        └── Main/         # train.txt / val.txt / test.txt split lists
    """
    # Create output directories
    voc_anno_dir = VOC_OUTPUT_ROOT / "Annotations"
    voc_img_dir = VOC_OUTPUT_ROOT / "JPEGImages"
    voc_sets_dir = VOC_OUTPUT_ROOT / "ImageSets" / "Main"
    voc_anno_dir.mkdir(parents=True, exist_ok=True)
    voc_img_dir.mkdir(parents=True, exist_ok=True)
    voc_sets_dir.mkdir(parents=True, exist_ok=True)

    total_img_count = 0
    total_obj_count = 0

    for split in SPLITS:
        print(f"Processing {split} set...")
        # Paths adapted to your directory structure
        yolo_img_dir = YOLO_ROOT / "images" / split
        yolo_label_dir = YOLO_ROOT / "labels" / split

        split_img_stems = []

        # Traverse all images in current split
        for img_path in yolo_img_dir.glob(f"*{IMAGE_EXT}"):
            img_name = img_path.name
            img_stem = img_path.stem
            split_img_stems.append(img_stem)

            # Copy original image to VOC image folder (no re-encoding)
            shutil.copy2(img_path, voc_img_dir / img_name)

            # Read image size and channel count
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                img_depth = len(img.getbands())

            # Read corresponding YOLO label file
            label_path = yolo_label_dir / f"{img_stem}.txt"
            objects = []

            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        # Parse YOLO normalized format: class_id x_center y_center width height
                        parts = line.split()
                        class_id = int(parts[0])
                        x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:5])

                        # Convert normalized coordinates to absolute pixel coordinates
                        x_center = x_center_norm * img_w
                        y_center = y_center_norm * img_h
                        box_w = w_norm * img_w
                        box_h = h_norm * img_h

                        # Calculate VOC-style (xmin, ymin, xmax, ymax)
                        xmin = max(0.0, x_center - box_w / 2)
                        ymin = max(0.0, y_center - box_h / 2)
                        xmax = min(float(img_w), x_center + box_w / 2)
                        ymax = min(float(img_h), y_center + box_h / 2)

                        # Skip invalid bounding boxes
                        if xmax <= xmin or ymax <= ymin:
                            continue

                        objects.append({
                            "name": CLASS_NAMES[class_id],
                            "bbox": (xmin, ymin, xmax, ymax)
                        })
                        total_obj_count += 1

            # Generate per-image XML annotation
            xml_path = voc_anno_dir / f"{img_stem}.xml"
            create_voc_xml(xml_path, img_name, (img_w, img_h, img_depth), objects)

            total_img_count += 1

        # Generate split index file
        split_list_file = voc_sets_dir / f"{split}.txt"
        with open(split_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(split_img_stems) + "\n")

        print(f"  {split} set completed: {len(split_img_stems)} images")

    # Print final summary
    print("\n" + "=" * 60)
    print("YOLO to VOC conversion finished successfully.")
    print(f"Total processed images: {total_img_count}")
    print(f"Total annotation objects: {total_obj_count}")
    print(f"Output directory: {VOC_OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    yolo_to_voc()