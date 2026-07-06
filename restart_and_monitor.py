import paramiko
import time

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect('192.168.20.88', username='enalis', password='pi', timeout=10)
        print("Connected to Pi!")
        
        print("Restarting strawmind.service (Flask Dashboard)...")
        stdin, stdout, stderr = ssh.exec_command("systemctl --user restart strawmind.service")
        err = stderr.read().decode('utf-8')
        if err:
            print(f"Error restarting strawmind: {err}")
        else:
            print("Flask Dashboard restarted successfully!")
            
        print("Restarting strawmind_yolo.service (YOLO Inference)...")
        stdin, stdout, stderr = ssh.exec_command("systemctl --user restart strawmind_yolo.service")
        err = stderr.read().decode('utf-8')
        if err:
            print(f"Error restarting strawmind_yolo: {err}")
        else:
            print("YOLO Inference restarted successfully!")
            
        print("Waiting 10 seconds for services to start and initialize...")
        time.sleep(10)
        
        print("\n--- Reading app.log (Flask App) ---")
        stdin, stdout, stderr = ssh.exec_command("tail -n 15 /home/enalis/strawmind/03_CODE/app.log")
        out = stdout.read().decode('utf-8').encode('ascii', 'ignore').decode('ascii')
        print(out)
        
        print("\n--- Reading yolo.log (YOLO Inference) ---")
        stdin, stdout, stderr = ssh.exec_command("tail -n 15 /home/enalis/strawmind/03_CODE/yolo.log")
        out = stdout.read().decode('utf-8').encode('ascii', 'ignore').decode('ascii')
        print(out)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
