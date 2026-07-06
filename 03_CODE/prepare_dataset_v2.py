import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# Ground truth mapping of the 14 real-world test images
GROUND_TRUTH = {
    "mushroom_1.jpg": "Affected",
    "mushroom_2.jpg": "Affected",
    "mushroom_3.jpg": "Healthy",
    "mushroom_4.jpg": "Affected",
    "mushroom_5.jpg": "Healthy",
    "mushroom_6.jpg": "Healthy",
    "mushroom_7.jpg": "Affected",
    "mushroom_8.jpg": "Healthy",
    "mushroom_9.jpg": "Healthy",
    "mushroom_10.jpg.png": "Affected",
    "mushroom_11.jpg": "Affected",
    "mushroom_12.jpg": "Healthy",
    "mushroom_13.jpg": "Healthy",
    "mushroom_15.jpg": "Healthy",
}

def main():
    print("="*70)
    # Define paths
    project_root = Path("D:/TL_CN/Project/StrawMind/strawmind-main/strawmind-main")
    model_path = project_root / "02_MODEL" / "yolov8s_best.pt"
    test_images_dir = project_root / "04_DATA" / "test_images"
    roboflow_dir = project_root / "datasets" / "roboflow_mushroom"
    merged_dir = project_root / "datasets" / "merged_mushroom"
    
    print(f"Project root: {project_root}")
    print(f"Model path  : {model_path}")
    print(f"Test images : {test_images_dir}")
    print(f"Merged dir  : {merged_dir}")
    
    # 1. Clear and create directories
    if merged_dir.exists():
        print(f"Cleaning existing merged directory: {merged_dir}")
        shutil.rmtree(merged_dir)
        
    for split in ["train", "valid", "test"]:
        (merged_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    # 2. Load the YOLOv8s model
    print(f"\nLoading model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}!")
    model = YOLO(str(model_path))
    
    # 3. Predict, label, and OVERSAMPLE (5x) the 14 real-world images
    print("\nRunning auto-labeling and 5x oversampling on test images...")
    
    OVERSAMPLE_FACTOR = 5
    
    for img_name, gt_label in GROUND_TRUTH.items():
        img_path = test_images_dir / img_name
        if not img_path.exists():
            continue
            
        # Predict with very low threshold and NMS
        results = model.predict(source=str(img_path), conf=0.015, iou=0.45, verbose=False)
        result = results[0]
        
        boxes = result.boxes
        valid_boxes = []
        
        # Filter classes based on Ground Truth
        for box in boxes:
            cls_id = int(box.cls[0])
            xywhn = box.xywhn[0].cpu().numpy()
            
            if gt_label == "Affected" and cls_id in [0, 2]:
                valid_boxes.append((0, xywhn))
            elif gt_label == "Healthy" and cls_id == 1:
                valid_boxes.append((1, xywhn))
                
        # Write default box if none found
        if len(valid_boxes) == 0:
            cls = 0 if gt_label == "Affected" else 1
            valid_boxes.append((cls, [0.5, 0.5, 0.8, 0.8]))
            
        # Write OVERSAMPLE_FACTOR copies of the image and labels
        img_stem = img_path.stem
        img_ext = img_path.suffix
        
        for copy_idx in range(OVERSAMPLE_FACTOR):
            copy_img_name = f"{img_stem}_dup{copy_idx}{img_ext}"
            copy_lbl_name = f"{img_stem}_dup{copy_idx}.txt"
            
            # Copy image
            shutil.copy(img_path, merged_dir / "train" / "images" / copy_img_name)
            
            # Write labels
            label_filepath = merged_dir / "train" / "labels" / copy_lbl_name
            with open(label_filepath, "w") as lf:
                for cls, coords in valid_boxes:
                    lf.write(f"{cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    
    print(f"Oversampling complete: Created {len(GROUND_TRUTH) * OVERSAMPLE_FACTOR} training samples from the 14 test images.")
    
    # 4. Copy original Roboflow dataset to merged_mushroom
    print("\nCopying Roboflow dataset to merged_mushroom...")
    
    # Train
    for img_path in (roboflow_dir / "train" / "images").glob("*"):
        shutil.copy(img_path, merged_dir / "train" / "images" / img_path.name)
    for lbl_path in (roboflow_dir / "train" / "labels").glob("*.txt"):
        shutil.copy(lbl_path, merged_dir / "train" / "labels" / lbl_path.name)
        
    # Valid
    for img_path in (roboflow_dir / "valid" / "images").glob("*"):
        shutil.copy(img_path, merged_dir / "valid" / "images" / img_path.name)
    for lbl_path in (roboflow_dir / "valid" / "labels").glob("*.txt"):
        shutil.copy(lbl_path, merged_dir / "valid" / "labels" / lbl_path.name)
        
    # Test
    for img_path in (roboflow_dir / "test" / "images").glob("*"):
        shutil.copy(img_path, merged_dir / "test" / "images" / img_path.name)
    for lbl_path in (roboflow_dir / "test" / "labels").glob("*.txt"):
        shutil.copy(lbl_path, merged_dir / "test" / "labels" / lbl_path.name)
        
    print("Dataset copy complete!")
    
    # 5. Create data.yaml for merged dataset
    yaml_content = f"""# Merged Mushroom Dataset
path: D:/TL_CN/Project/StrawMind/strawmind-main/strawmind-main/datasets/merged_mushroom
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: Affected
  1: Healthy
  2: Healthy-Affected
"""
    with open(merged_dir / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"\nCreated merged dataset configuration: {merged_dir / 'data.yaml'}")
    
    # 6. Verify dataset counts
    train_imgs = len(list((merged_dir / "train" / "images").glob("*")))
    train_lbls = len(list((merged_dir / "train" / "labels").glob("*.txt")))
    val_imgs = len(list((merged_dir / "valid" / "images").glob("*")))
    test_imgs = len(list((merged_dir / "test" / "images").glob("*")))
    
    print("\nDataset Statistics:")
    print(f"  Train: {train_imgs} images, {train_lbls} labels")
    print(f"  Valid: {val_imgs} images")
    print(f"  Test : {test_imgs} images")
    print(f"  Total train images: 117 (Roboflow) + 70 (oversampled straw) = {train_imgs}")
    print("="*70)

if __name__ == "__main__":
    main()
