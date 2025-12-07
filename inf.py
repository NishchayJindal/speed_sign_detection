import os
import time
import io
import csv
import argparse
from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import yaml

# ================== PATH CONFIG ==================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.yaml"

# ================== MODEL PATH ==================

MODEL_PATH = r"C:\Users\Dell\Desktop\best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ best.pt not found at: {MODEL_PATH}")

CONF_DEFAULT = 0.81
IMG_SIZE_DEFAULT = 640
LIMIT_HOLD_SEC = 5.0


# ================== HELPERS ==================

def load_model_safe(path: str):
    return YOLO(path)


def load_class_names(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    return {i: n for i, n in enumerate(names)} if isinstance(names, list) else names


def build_speed_limit_mapping(class_names: dict):
    mapping = {}
    for cid, name in class_names.items():
        low = name.lower()
        if "speed limit" in low and any(ch.isdigit() for ch in low):
            nums = [p for p in low.replace("/", " ").split() if p.isdigit()]
            if nums:
                mapping[cid] = float(nums[-1])
    return mapping


def classify_turn_sign(cls_name_lower: str):
    if "no u-turn" in cls_name_lower or "no u turn" in cls_name_lower:
        return "no_uturn"
    if "no left" in cls_name_lower:
        return "no_left"
    if "no right" in cls_name_lower:
        return "no_right"
    return None


# ================== PROCESS FRAME ==================

def process_frame_with_violations(
    model, frame_bgr, class_names, speed_limit_classes, vehicle_speed, car_moving,
    manoeuvre, last_limit_value, last_limit_time, conf, imgsz,
):

    results = model.predict(frame_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]

    vis = frame_bgr.copy()
    h, w, _ = vis.shape
    any_violation = False
    violated_msgs = []
    detected_signs = []

    man = manoeuvre.lower()
    is_parked = "parked" in man
    turning_left = "turning left" in man
    turning_right = "turning right" in man
    doing_uturn = "u-turn" in man or "u turn" in man

    current_limit = None
    best_conf = 0

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf_box = float(box.conf[0])
            cls_name = class_names.get(cls, str(cls))
            cls_low = cls_name.lower()

            detected_signs.append(cls_name)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{cls_name} {conf_box*100:.1f}%", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            # SPEED LIMIT
            if cls in speed_limit_classes:
                limit = speed_limit_classes[cls]
                if conf_box > best_conf:
                    best_conf = conf_box
                    current_limit = limit
                if vehicle_speed > limit + 2:
                    any_violation = True
                    violated_msgs.append(f"Over Speed {int(limit)}")

            # RED LIGHT
            if cls_low == "red light" and car_moving:
                any_violation = True
                violated_msgs.append("Red Light Violation")

            # STOP SIGN
            if cls_low == "stop" and car_moving:
                any_violation = True
                violated_msgs.append("Stop Sign Violation")

            # NO PARKING
            if "no parking" in cls_low and is_parked:
                any_violation = True
                violated_msgs.append("No Parking Violation")

            # TURN SIGNS
            turn = classify_turn_sign(cls_low)
            if turn == "no_left" and turning_left:
                any_violation = True
                violated_msgs.append("No Left Turn Violation")
            if turn == "no_right" and turning_right:
                any_violation = True
                violated_msgs.append("No Right Turn Violation")
            if turn == "no_uturn" and doing_uturn:
                any_violation = True
                violated_msgs.append("No U-Turn Violation")

    # Limit memory
    now = time.time()
    if current_limit is not None:
        last_limit_value, last_limit_time = current_limit, now
    elif last_limit_value and now - last_limit_time > LIMIT_HOLD_SEC:
        last_limit_value = None

    cv2.putText(vis,f"Speed: {vehicle_speed:.1f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    if last_limit_value is not None:
        cv2.putText(vis,f"Limit: {int(last_limit_value)}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    if any_violation:
        cv2.rectangle(vis,(0,0),(w,70),(0,0,255),-1)
        cv2.putText(vis,"SIGN VIOLATED",(20,50),cv2.FONT_HERSHEY_DUPLEX,1.3,(255,255,255),3)

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), last_limit_value, last_limit_time, any_violation, violated_msgs, detected_signs


# ================== VIDEO INFERENCE ==================

def infer_video_with_violations(
    model, video_path, output_video_path, output_csv_path, class_names,
    speed_limit_classes, vehicle_speed, car_moving, manoeuvre, conf, imgsz,
):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    last_limit, last_time = None, 0
    sign_counter = Counter()
    frame = 0

    # ========== NEW VIOLATION TRACKING SYSTEM ==========
    active_violations = {}      # {"Red Light": {start: t, signs:set()}}
    completed_events = []       # Final Violation Log

    while True:
        ret, frm = cap.read()
        if not ret: break
        frame += 1

        annotated, last_limit, last_time, any_v, msgs, signs = process_frame_with_violations(
            model, frm, class_names, speed_limit_classes, vehicle_speed,
            car_moving, manoeuvre, last_limit, last_time, conf, imgsz,
        )

        writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))
        sign_counter.update(signs)

        timestamp = frame / fps

        # ========== START / CONTINUE VIOLATION ==========
        if any_v:
            for v in msgs:
                if v not in active_violations:
                    active_violations[v] = {"start": timestamp, "signs": set(signs)}
                else:
                    active_violations[v]["signs"].update(signs)

            # End violation that is no longer active
            for v in list(active_violations):
                if v not in msgs:
                    ev = active_violations.pop(v)
                    completed_events.append({
                        "violation": v,
                        "start_time": round(ev["start"],2),
                        "end_time": round(timestamp,2),
                        "duration_sec": round(timestamp-ev["start"],2),
                        "vehicle_speed": vehicle_speed,
                        "manoeuvre": manoeuvre,
                        "detected_signs": ";".join(ev["signs"])
                    })

        else:  # ended all violations
            for v,ev in active_violations.items():
                completed_events.append({
                    "violation": v,
                    "start_time": round(ev["start"],2),
                    "end_time": round(timestamp,2),
                    "duration_sec": round(timestamp-ev["start"],2),
                    "vehicle_speed": vehicle_speed,
                    "manoeuvre": manoeuvre,
                    "detected_signs": ";".join(ev["signs"])
                })
            active_violations.clear()

    cap.release()
    writer.release()

    # ========== SAVE CSV ==========
    fieldnames = ["violation","start_time","end_time","duration_sec",
                  "vehicle_speed","manoeuvre","detected_signs"]

    with open(output_csv_path,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed_events)

    print("DONE — CSV with start & end timestamps saved:", output_csv_path)
    print("Annotated video saved:", output_video_path)
    print("Summary:", dict(sign_counter))

    return output_video_path, output_csv_path, frame, sign_counter


# ================== MAIN ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_video", default="annotated.mp4")
    parser.add_argument("--output_csv", default="violations.csv")
    parser.add_argument("--conf", type=float, default=CONF_DEFAULT)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE_DEFAULT)
    parser.add_argument("--vehicle_speed", type=float, default=50.0)
    parser.add_argument("--car_moving", type=int, default=1)
    parser.add_argument("--manoeuvre", type=str, default="Driving straight",
                        choices=["Driving straight","Parked","Turning left","Turning right","U-turn"])
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError("data.yaml missing!")

    class_names = load_class_names(DATA_PATH)
    speed_map = build_speed_limit_mapping(class_names)

    print("Loading YOLO model...")
    model = load_model_safe(MODEL_PATH)
    print("Model Loaded!")

    infer_video_with_violations(
        model, args.video, args.output_video, args.output_csv,
        class_names, speed_map, args.vehicle_speed, bool(args.car_moving),
        args.manoeuvre, args.conf, args.imgsz
    )

if __name__ == "__main__":
    main()
