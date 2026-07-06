import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
from pathlib import Path

def main():
    print("="*70)
    print("STRAWMIND - PRUNING YOLOv8 NANO MODEL")
    print("="*70)
    
    project_root = Path("D:/TL_CN/Project/StrawMind/strawmind-main/strawmind-main")
    model_path = project_root / 'runs' / 'train_yolov8n_320' / 'strawmind_yolov8n_320' / 'weights' / 'best.pt'
    output_path = project_root / 'runs' / 'train_yolov8n_320' / 'strawmind_yolov8n_320' / 'weights' / 'best_pruned.pt'
    
    if not model_path.exists():
        print(f"ERROR: Model weights not found at {model_path}!")
        print("Please wait for the training process to complete.")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    pytorch_model = model.model
    
    print("\nApplying L1-unstructured pruning to all Conv2d layers (30% sparsity)...")
    
    conv_layers_count = 0
    total_params = 0
    pruned_params = 0
    
    for name, module in pytorch_model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name.startswith('model.22'):
                print(f"Skipping pruning for detection head layer: {name}")
                continue
            conv_layers_count += 1
            # Calculate parameters in this layer
            weight = module.weight.data
            layer_params = weight.numel()
            total_params += layer_params
            
            # Apply L1 unstructured pruning
            prune.l1_unstructured(module, name='weight', amount=0.3)
            # Remove pruning hook to make the zero weights permanent
            prune.remove(module, 'weight')
            
            # Count actual zeros in pruned weight
            zeros = torch.sum(module.weight.data == 0).item()
            pruned_params += zeros
            
    print(f"\nPruning Summary:")
    print(f"  Total Conv2d layers pruned: {conv_layers_count}")
    print(f"  Total Conv2d parameters   : {total_params:,}")
    print(f"  Pruned parameters (zeros) : {pruned_params:,}")
    print(f"  Actual Sparsity           : {pruned_params / total_params * 100:.2f}%")
    
    # Save the pruned model
    print(f"\nSaving pruned model to: {output_path}")
    model.save(str(output_path))
    print("Pruned model saved successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
