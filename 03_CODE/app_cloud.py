import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, render_template, request, send_file

try:
    from supabase import create_client
except Exception:
    create_client = None

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "02_MODEL" / "yolov8n_best.onnx"))
DATA_DIR = ROOT_DIR / "04_DATA"
SNAPSHOT_DIR = DATA_DIR / "disease_snapshots"
DISEASE_LOGS_FILE = DATA_DIR / "disease_logs.json"
LATEST_RAW_PATH = DATA_DIR / "cloud_latest.jpg"
LATEST_ANALYZED_PATH = DATA_DIR / "cloud_analyzed.jpg"

MQTT_HOST = os.getenv("MQTT_HOST", "")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "strawmind")
NODE_ID = os.getenv("NODE_ID", "bed-01")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
if not DISEASE_LOGS_FILE.exists():
    DISEASE_LOGS_FILE.write_text("[]", encoding="utf-8")

app = Flask(__name__)
data_lock = threading.Lock()
log_lock = threading.Lock()
model_lock = threading.Lock()

sensor_data = {
    "temp": 0,
    "hum": 0,
    "eco2": 0,
    "aqi": 0,
    "substrateMoisture": 0,
    "fan": 0,
    "fogger": 0,
    "mode": "manual",
    "status": "Đang chờ MQTT telemetry",
    "healthy_count": 0,
    "affected_count": 0,
    "disease_detected": False,
    "analysis_time": 0,
    "last_updated": None,
}

mqtt_status = {
    "enabled": bool(MQTT_HOST and MQTT_USERNAME and MQTT_PASSWORD),
    "connected": False,
    "last_error": None,
    "last_message_at": None,
}

supabase_status = {
    "enabled": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client),
    "last_error": None,
    "last_write_at": None,
}

supabase = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    if supabase_status["enabled"]
    else None
)

mqtt_client = None
device_id_cache = {}


class YOLOv8ONNX:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = {0: "Affected Mushroom", 1: "Healthy Mushroom", 2: "Affected Mushroom"}

    def __call__(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        pred = np.transpose(preds[0], (1, 0))

        boxes, confidences, class_ids = [], [], []
        img_h, img_w = frame.shape[:2]
        x_factor = img_w / 320.0
        y_factor = img_h / 320.0

        for row in pred:
            confidence = float(np.max(row[4:]))
            if confidence < 0.012:
                continue
            class_id = int(np.argmax(row[4:]))
            x_center, y_center, w, h = row[:4]
            x = int((x_center - w / 2) * x_factor)
            y = int((y_center - h / 2) * y_factor)
            boxes.append([x, y, int(w * x_factor), int(h * y_factor)])
            confidences.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.012, 0.45)
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten() if hasattr(indices, "flatten") else indices:
                x, y, w, h = boxes[idx]
                label = self.names.get(class_ids[idx], "Mushroom")
                detections.append(
                    {
                        "box": [x, y, x + w, y + h],
                        "label": label,
                        "conf": float(confidences[idx]),
                    }
                )
        return detections


model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = YOLOv8ONNX(MODEL_PATH)
    return model


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def map_telemetry(payload):
    return {
        "temp": float(payload.get("temperature", payload.get("temp", 0)) or 0),
        "hum": float(payload.get("humidity", payload.get("hum", 0)) or 0),
        "eco2": int(payload.get("co2", payload.get("eco2", 0)) or 0),
        "aqi": int(payload.get("aqi", 0) or 0),
        "substrateMoisture": float(payload.get("substrateMoisture", 0) or 0),
        "fan": int(payload.get("fan", 0) or 0),
        "fogger": int(payload.get("fogger", payload.get("mist", 0)) or 0),
        "mode": str(payload.get("mode", sensor_data["mode"])),
        "status": "Đang kết nối qua HiveMQ",
        "last_updated": now_iso(),
    }


def get_or_create_device(node_id):
    if not supabase:
        return None
    if node_id in device_id_cache:
        return device_id_cache[node_id]

    existing = (
        supabase.table("devices").select("id").eq("device_code", node_id).execute().data
    )
    if existing:
        device_id_cache[node_id] = existing[0]["id"]
        return existing[0]["id"]

    created = (
        supabase.table("devices")
        .insert({"device_code": node_id, "name": node_id, "type": "iot_node", "status": "online"})
        .execute()
        .data
    )
    device_id_cache[node_id] = created[0]["id"]
    return created[0]["id"]


def persist_sensor(payload):
    if not supabase:
        return
    try:
        device_id = get_or_create_device(NODE_ID)
        supabase.table("devices").update({"status": "online", "last_seen_at": now_iso()}).eq("id", device_id).execute()
        supabase.table("sensor_logs").insert(
            {
                "device_id": device_id,
                "temperature": payload["temp"],
                "humidity": payload["hum"],
                "co2": payload["eco2"],
                "soil_moisture": payload["substrateMoisture"],
            }
        ).execute()
        supabase_status["last_error"] = None
        supabase_status["last_write_at"] = now_iso()
    except Exception as exc:
        supabase_status["last_error"] = str(exc)


def persist_detection(detections, image_url=None, inference_time_ms=None):
    if not supabase:
        return
    try:
        device_id = get_or_create_device(NODE_ID)
        image = (
            supabase.table("mushroom_images")
            .insert({"device_id": device_id, "image_url": image_url or "", "captured_at": now_iso()})
            .execute()
            .data[0]
        )
        if detections:
            supabase.table("ai_detections").insert(
                [
                    {
                        "image_id": image["id"],
                        "model_name": "YOLOv8n-ONNX-Cloud",
                        "class_name": item["label"],
                        "confidence": item["conf"],
                        "bbox": {
                            "x1": item["box"][0],
                            "y1": item["box"][1],
                            "x2": item["box"][2],
                            "y2": item["box"][3],
                        },
                        "inference_time_ms": inference_time_ms,
                    }
                    for item in detections
                ]
            ).execute()
        supabase_status["last_error"] = None
        supabase_status["last_write_at"] = now_iso()
    except Exception as exc:
        supabase_status["last_error"] = str(exc)


def append_disease_log(affected_count, detections):
    if affected_count <= 0:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"snapshot_{timestamp}.jpg"
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    if LATEST_ANALYZED_PATH.exists():
        snapshot_path.write_bytes(LATEST_ANALYZED_PATH.read_bytes())

    log = {
        "id": f"log_{int(time.time())}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease_type": "Affected Mushroom (Nấm nhiễm bệnh)",
        "count": affected_count,
        "temp": sensor_data.get("temp", 0),
        "hum": sensor_data.get("hum", 0),
        "image_url": f"/disease_snapshot/{snapshot_name}",
        "recommendations": [
            "Cách ly vùng nấm nghi nhiễm bệnh và kiểm tra độ ẩm/thông gió.",
            "Theo dõi thêm các khung hình kế tiếp để xác nhận mức độ lan rộng.",
        ],
        "detections": detections,
    }

    with log_lock:
        try:
            logs = json.loads(DISEASE_LOGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            logs = []
        logs.insert(0, log)
        DISEASE_LOGS_FILE.write_text(json.dumps(logs[:50], ensure_ascii=False, indent=2), encoding="utf-8")


def on_mqtt_connect(client, _userdata, _flags, rc):
    mqtt_status["connected"] = rc == 0
    if rc != 0:
        mqtt_status["last_error"] = f"MQTT connect rc={rc}"
        return
    topic = f"{MQTT_TOPIC_PREFIX}/+/telemetry"
    client.subscribe(topic, qos=1)
    mqtt_status["last_error"] = None


def on_mqtt_message(_client, _userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        mapped = map_telemetry(payload)
        with data_lock:
            sensor_data.update(mapped)
        mqtt_status["last_message_at"] = now_iso()
        persist_sensor(mapped)
    except Exception as exc:
        mqtt_status["last_error"] = str(exc)


def start_mqtt():
    global mqtt_client
    if not mqtt_status["enabled"]:
        return
    mqtt_client = mqtt.Client(client_id=f"strawmind-core-cloud-{os.getpid()}")
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.tls_set()
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()


def publish_command(actuator, state):
    if not mqtt_client or not mqtt_status["connected"]:
        return False
    topic = f"{MQTT_TOPIC_PREFIX}/{NODE_ID}/cmd/{actuator}"
    payload = json.dumps({"state": "on" if state else "off"})
    result = mqtt_client.publish(topic, payload, qos=1)
    return result.rc == mqtt.MQTT_ERR_SUCCESS


def draw_detections(frame, detections):
    output = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        conf = item["conf"]
        color = (0, 200, 0) if "Healthy" in label else (0, 0, 220)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output


def placeholder_frame(text="Waiting for cloud image upload"):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 35, 28)
    cv2.putText(frame, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
    return frame


def jpeg_response(frame):
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        return Response(status=500)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    with data_lock:
        return jsonify(sensor_data)


@app.route("/api/status")
def get_status():
    return jsonify({"mqtt": mqtt_status, "supabase": supabase_status, "model": str(MODEL_PATH)})


@app.route("/api/control", methods=["POST"])
def control():
    payload = request.get_json(silent=True) or {}
    with data_lock:
        if "mode" in payload:
            sensor_data["mode"] = payload["mode"]
        if "fan" in payload:
            sensor_data["fan"] = int(payload["fan"])
            publish_command("fan", bool(payload["fan"]))
        if "fogger" in payload:
            sensor_data["fogger"] = int(payload["fogger"])
            publish_command("mist", bool(payload["fogger"]))
    return jsonify(sensor_data)


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    return jsonify(
        {
            "cam_ip": "cloud-upload",
            "sensor_ip": "hivemq-cloud",
            "cam_simulation": True,
            "sensor_simulation": not mqtt_status["connected"],
            "mqtt": mqtt_status,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Upload multipart field named image"}), 400
    file = request.files["image"]
    raw = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    start = time.time()
    with model_lock:
        detections = load_model()(frame)
    inference_time_ms = int((time.time() - start) * 1000)

    analyzed = draw_detections(frame, detections)
    cv2.imwrite(str(LATEST_RAW_PATH), frame)
    cv2.imwrite(str(LATEST_ANALYZED_PATH), analyzed)

    healthy_count = sum(1 for item in detections if "Healthy" in item["label"])
    affected_count = sum(1 for item in detections if "Affected" in item["label"])
    with data_lock:
        sensor_data.update(
            {
                "healthy_count": healthy_count,
                "affected_count": affected_count,
                "disease_detected": affected_count > 0,
                "analysis_time": time.time(),
            }
        )

    append_disease_log(affected_count, detections)
    persist_detection(detections, image_url="/analyzed_image", inference_time_ms=inference_time_ms)

    return jsonify(
        {
            "healthy_count": healthy_count,
            "affected_count": affected_count,
            "detections": detections,
            "inference_time_ms": inference_time_ms,
            "analyzed_image": "/analyzed_image",
        }
    )


@app.route("/api/yolo_update", methods=["POST"])
def yolo_update():
    payload = request.get_json(silent=True) or {}
    boxes = payload.get("boxes", [])
    healthy_count = int(payload.get("healthy_count", 0))
    affected_count = int(payload.get("affected_count", 0))
    with data_lock:
        sensor_data.update(
            {
                "healthy_count": healthy_count,
                "affected_count": affected_count,
                "disease_detected": affected_count > 0,
                "analysis_time": time.time(),
            }
        )
    append_disease_log(affected_count, boxes)
    return jsonify({"status": "success"})


@app.route("/api/disease_logs")
def disease_logs():
    try:
        return jsonify(json.loads(DISEASE_LOGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        return jsonify([])


@app.route("/disease_snapshot/<filename>")
def disease_snapshot(filename):
    path = SNAPSHOT_DIR / filename
    if path.exists():
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "not found"}), 404


@app.route("/analyzed_image")
def analyzed_image():
    if LATEST_ANALYZED_PATH.exists():
        return send_file(LATEST_ANALYZED_PATH, mimetype="image/jpeg")
    return jpeg_response(placeholder_frame("Waiting for AI analysis"))


@app.route("/video_feed")
def video_feed():
    def stream():
        while True:
            if LATEST_RAW_PATH.exists():
                frame = cv2.imread(str(LATEST_RAW_PATH))
            else:
                frame = placeholder_frame("Cloud mode: upload an image to /api/analyze")
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(1)

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


start_mqtt()

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
