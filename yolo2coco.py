import json
import shutil
from pathlib import Path
from PIL import Image

# ============================================================
# Root directory of the YOLO dataset (includes images, labels and classes.txt)
YOLO_ROOT = Path("/LGPSD-DET")
# Output directory for the converted COCO dataset
COCO_OUTPUT_ROOT = Path("/LGPSD-DET_COCO")
# Path to the class definition file
CLASSES_FILE = YOLO_ROOT / "classes.txt"
# Image file extension in your dataset
IMAGE_EXT = ".png"
# Dataset splits to process
SPLITS = ["train", "val", "test"]

# Automatically load class names from classes.txt (index order matches YOLO labels)
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]


def yolo_to_coco():
    """
    Main conversion function.
    Output standard COCO directory structure:
    LGPSD-DET_COCO/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
    """
    # Create output directories
    coco_img_root = COCO_OUTPUT_ROOT / "images"
    coco_anno_dir = COCO_OUTPUT_ROOT / "annotations"
    coco_anno_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (coco_img_root / split).mkdir(parents=True, exist_ok=True)

    # Global unique annotation ID (increment across all splits)
    annotation_id = 1

    for split in SPLITS:
        print(f"Processing {split} set...")
        # Paths adapted to your directory structure
        yolo_img_dir = YOLO_ROOT / "images" / split
        yolo_label_dir = YOLO_ROOT / "labels" / split

        # Initialize standard COCO JSON structure
        coco_data = {
            "info": {
                "description": "LGPSD-DET: Light Guide Plate Surface Defect Detection Dataset",
                "version": "1.0",
                "year": 2026,
                "date_created": "2026-08-14"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "CC BY 4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Fill category list (COCO convention: category IDs start from 1)
        for idx, cat_name in enumerate(CLASS_NAMES):
            coco_data["categories"].append({
                "id": idx + 1,
                "name": cat_name,
                "supercategory": "defect"
            })

        image_id = 1
        # Process images in sorted order for reproducibility
        for img_path in sorted(yolo_img_dir.glob(f"*{IMAGE_EXT}")):
            img_name = img_path.name

            # Copy original image to COCO image folder (no re-encoding)
            shutil.copy2(img_path, coco_img_root / split / img_name)

            # Read image dimensions
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # Add image entry to COCO dict
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": img_w,
                "height": img_h,
                "license": 1
            })

            # Read and parse corresponding YOLO label file
            label_path = yolo_label_dir / f"{img_path.stem}.txt"
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

                        # Convert to COCO bbox format: [x_top_left, y_top_left, width, height]
                        box_w = w_norm * img_w
                        box_h = h_norm * img_h
                        x_top_left = max(0.0, x_center_norm * img_w - box_w / 2)
                        y_top_left = max(0.0, y_center_norm * img_h - box_h / 2)

                        # Clip bounding box to image boundaries
                        box_w = min(box_w, img_w - x_top_left)
                        box_h = min(box_h, img_h - y_top_left)

                        # Skip invalid boxes
                        if box_w <= 0 or box_h <= 0:
                            continue

                        # Calculate bounding box area
                        box_area = box_w * box_h

                        # Add annotation entry
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,
                            "bbox": [x_top_left, y_top_left, box_w, box_h],
                            "area": box_area,
                            "iscrowd": 0,
                            "segmentation": []
                        })
                        annotation_id += 1

            image_id += 1

        # Save COCO JSON annotation file
        json_output_path = coco_anno_dir / f"instances_{split}.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"  {split} set completed: {image_id - 1} images")

    # Print final summary
    print("\n" + "=" * 60)
    print("YOLO to COCO conversion finished successfully.")
    print(f"Total generated annotations: {annotation_id - 1}")
    print(f"Output directory: {COCO_OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    yolo_to_coco()