import os
import paramiko
from pathlib import Path

def main():
    print("="*70)
    print("STRAWMIND - DEPLOYING NEW YOLOv8 NANO MODEL AND SERVICE TO PI 3")
    print("="*70)
    
    # Define local files
    project_root = Path("D:/TL_CN/Project/StrawMind/strawmind-main/strawmind-main")
    local_model = project_root / 'runs' / 'train_yolov8n_320' / 'strawmind_yolov8n_320' / 'weights' / 'best_pruned.onnx'
    local_service = project_root / '03_CODE' / 'yolo_service.py'
    local_app = project_root / '03_CODE' / 'app.py'
    local_index = project_root / '03_CODE' / 'templates' / 'index.html'
    
    # Define remote paths
    remote_model = "/home/enalis/strawmind/02_MODEL/yolov8n_best.onnx"
    remote_service = "/home/enalis/strawmind/03_CODE/yolo_service.py"
    remote_app = "/home/enalis/strawmind/03_CODE/app.py"
    remote_index = "/home/enalis/strawmind/03_CODE/templates/index.html"
    
    # Connection details
    pi_ip = "192.168.20.88"
    pi_user = "enalis"
    pi_pass = "pi"
    
    print("Checking local files...")
    if not local_model.exists():
        raise FileNotFoundError(f"Local model not found: {local_model}")
    if not local_service.exists():
        raise FileNotFoundError(f"Local service file not found: {local_service}")
    if not local_app.exists():
        raise FileNotFoundError(f"Local app file not found: {local_app}")
    if not local_index.exists():
        raise FileNotFoundError(f"Local index file not found: {local_index}")
        
    print(f"  Model size  : {os.path.getsize(local_model) / (1024*1024):.2f} MB")
    print(f"  Service size: {os.path.getsize(local_service) / 1024:.2f} KB")
    print(f"  App size    : {os.path.getsize(local_app) / 1024:.2f} KB")
    print(f"  Index size  : {os.path.getsize(local_index) / 1024:.2f} KB")
    
    # Establish SFTP Connection
    print(f"\nConnecting to Pi at {pi_ip}...")
    transport = paramiko.Transport((pi_ip, 22))
    try:
        transport.connect(username=pi_user, password=pi_pass)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("Connected!")
        
        # Ensure directories exist
        try:
            sftp.mkdir("/home/enalis/strawmind/02_MODEL")
            print("Created remote dir: 02_MODEL")
        except IOError:
            pass
            
        try:
            sftp.mkdir("/home/enalis/strawmind/03_CODE")
            print("Created remote dir: 03_CODE")
        except IOError:
            pass
            
        try:
            sftp.mkdir("/home/enalis/strawmind/03_CODE/templates")
            print("Created remote dir: 03_CODE/templates")
        except IOError:
            pass
            
        # Uploading model
        print(f"\nUploading model: {local_model} -> {remote_model}...")
        sftp.put(str(local_model), remote_model)
        print("Model uploaded successfully!")
        
        # Uploading service
        print(f"\nUploading service script: {local_service} -> {remote_service}...")
        sftp.put(str(local_service), remote_service)
        print("Service script uploaded successfully!")
        
        # Uploading Flask app
        print(f"\nUploading Flask app: {local_app} -> {remote_app}...")
        sftp.put(str(local_app), remote_app)
        print("Flask app uploaded successfully!")
        
        # Uploading HTML index
        print(f"\nUploading HTML template: {local_index} -> {remote_index}...")
        sftp.put(str(local_index), remote_index)
        print("HTML template uploaded successfully!")
        
    except Exception as e:
        print(f"Error during deployment: {e}")
    finally:
        transport.close()
        print("\nConnection closed.")
    print("="*70)

if __name__ == "__main__":
    main()
