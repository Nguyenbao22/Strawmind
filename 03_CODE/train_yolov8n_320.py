import os
import time
from pathlib import Path
from ultralytics import YOLO
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    print("="*70)
    print("STRAWMIND - TRAINING YOLOV8 NANO AT 320x320 RESOLUTION")
    print("="*70)
    
    # Paths
    project_root = Path("D:/TL_CN/Project/StrawMind/strawmind-main/strawmind-main")
    data_yaml = project_root / "datasets" / "merged_mushroom" / "data.yaml"
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device selected: {device}")
    
    # Load pretrained YOLOv8n model
    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Pretrained YOLOv8n loaded successfully!")
    
    # Training configuration
    config = {
        'data': str(data_yaml),
        'imgsz': 320,
        'epochs': 100,
        'patience': 30,          # Early stopping patience
        'batch': 8,              # Optimized batch size for CPU/light GPU
        'device': device,
        'workers': 2,
        'project': str(project_root / 'runs' / 'train_yolov8n_320'),
        'name': 'strawmind_yolov8n_320',
        'save': True,
        'plots': True,
        'verbose': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'cos_lr': True,          # Cosine learning rate scheduler
        'amp': False,            # Disable AMP for CPU stability
    }
    
    print("\nTraining Parameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
        
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        results = model.train(**config)
        elapsed = time.time() - start_time
        h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print(f"Time elapsed: {h}h {m}m")
        
        best_weights = project_root / 'runs' / 'train_yolov8n_320' / 'strawmind_yolov8n_320' / 'weights' / 'best.pt'
        print(f"Best model weights saved at: {best_weights}")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
