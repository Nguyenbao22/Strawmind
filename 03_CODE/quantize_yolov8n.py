import os
from pathlib import Path
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    print("="*70)
    print("STRAWMIND - EXPORTING AND QUANTIZING MODEL TO ONNX INT8")
    print("="*70)
    
    project_root = Path("D:/TL_CN/Project/StrawMind/strawmind-main/strawmind-main")
    pruned_model_path = project_root / 'runs' / 'train_yolov8n_320' / 'strawmind_yolov8n_320' / 'weights' / 'best_pruned.pt'
    
    if not pruned_model_path.exists():
        print(f"ERROR: Pruned model not found at {pruned_model_path}!")
        print("Please run prune_yolov8n.py first.")
        return
        
    # 1. Load the pruned PyTorch model
    print(f"Loading pruned model: {pruned_model_path}")
    model = YOLO(str(pruned_model_path))
    
    # 2. Export to ONNX at 320x320 static resolution
    print("\nExporting model to ONNX (320x320)...")
    onnx_file_path = model.export(format="onnx", imgsz=320)
    print(f"Exported FP32 ONNX model saved at: {onnx_file_path}")
    
    # Verify FP32 size
    fp32_size = os.path.getsize(onnx_file_path) / (1024 * 1024)
    print(f"FP32 ONNX model size: {fp32_size:.2f} MB")
    
    # 3. Dynamic Quantization to INT8
    print("\nQuantizing ONNX model to INT8 (Dynamic)...")
    quantized_output_path = pruned_model_path.parent / "yolov8n_pruned_quantized.onnx"
    
    quantize_dynamic(
        model_input=str(onnx_file_path),
        model_output=str(quantized_output_path),
        weight_type=QuantType.QInt8
    )
    
    # Verify INT8 size
    int8_size = os.path.getsize(quantized_output_path) / (1024 * 1024)
    print(f"\nQuantized INT8 ONNX model saved at: {quantized_output_path}")
    print(f"INT8 ONNX model size: {int8_size:.2f} MB (Target: ~3MB)")
    print(f"Size reduction ratio: {fp32_size / int8_size:.2f}x smaller!")
    print("="*70)

if __name__ == "__main__":
    main()
