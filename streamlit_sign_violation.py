import streamlit as st
from ultralytics import YOLO
import tempfile, os, time, io
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import yaml
from collections import Counter

st.set_page_config(page_title="Traffic Sign Violation (YOLO)", layout="wide")

# ================== PATH CONFIG ==================

BASE_DIR = Path(__file__).resolve().parent
MODEL_SEARCH_DIR = BASE_DIR / "runs" / "detect"
DATA_PATH = BASE_DIR / "data.yaml"
DEFAULT_MODEL_FILE = BASE_DIR / "yolov8s.pt"  # fallback if no run weights

CONF_DEFAULT = 0.4
IMG_SIZE_DEFAULT = 640
LIMIT_HOLD_SEC = 5.0  # remember last seen speed limit


# ================== HELPERS ==================

def find_run_weights(search_dir: Path):
    """Scan runs/detect/*/weights/*.pt and return (display_name, full_path)."""
    res = []
    if not search_dir.exists():
        return res
    for run_dir in search_dir.iterdir():
        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            for pt in weights_dir.glob("*.pt"):
                display = f"{run_dir.name}/weights/{pt.name}"
                res.append((display, str(pt.resolve()), run_dir.stat().st_mtime))
    res.sort(key=lambda x: x[2], reverse=True)
    return [(d, p) for (d, p, _) in res]


@st.cache_resource
def load_model_safe(path: str):
    return YOLO(path)


@st.cache_resource
def load_class_names(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return names


@st.cache_resource
def build_speed_limit_mapping(class_names: dict):
    """class_id -> numeric speed from labels like 'Speed Limit 40'."""
    mapping = {}
    for cid, name in class_names.items():
        if not isinstance(name, str):
            continue
        low = name.lower()
        if "speed limit" in low and any(ch.isdigit() for ch in low):
            parts = low.replace("/", " ").split()
            nums = [p for p in parts if p.isdigit()]
            if nums:
                mapping[cid] = float(nums[-1])
    return mapping


def classify_turn_sign(cls_name_lower: str):
    """
    Roughly classify turn-related prohibitory signs based on name.
    This only triggers if your future dataset actually has these names.
    """
    if "no u-turn" in cls_name_lower or "no u turn" in cls_name_lower or "u-turn prohibited" in cls_name_lower:
        return "no_uturn"
    if "no left" in cls_name_lower or "no left turn" in cls_name_lower:
        return "no_left"
    if "no right" in cls_name_lower or "no right turn" in cls_name_lower:
        return "no_right"
    return None


def process_frame_with_violations(
    model,
    frame_bgr,
    class_names,
    speed_limit_classes,
    vehicle_speed: float,
    car_moving: bool,
    manoeuvre: str,
    last_limit_value,
    last_limit_time,
    conf: float,
    imgsz: int,
):
    """
    Run YOLO on a BGR frame, draw bboxes + labels, and apply rules:
      - Speed limit
      - Red light
      - Stop
      - (Optionally) No Parking, No Left/Right/U-Turn if those classes exist

    Returns:
      pil_rgb_frame,
      new_last_limit_value,
      new_last_limit_time,
      any_violation,
      violation_msgs,
      detected_signs  (list of sign names in this frame)
    """
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )
    result = results[0]

    vis = frame_bgr.copy()
    h, w, _ = vis.shape

    any_violation = False
    violated_msgs = []
    detected_signs = []

    current_frame_limit = None
    best_limit_conf = 0.0

    # interpret manoeuvre
    man_lower = manoeuvre.lower()
    is_parked = "parked" in man_lower or "standing" in man_lower
    turning_left = "turning left" in man_lower
    turning_right = "turning right" in man_lower
    doing_uturn = "u-turn" in man_lower or "u turn" in man_lower

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(box.cls[0])
            conf_box = float(box.conf[0])
            cls_name = class_names.get(cls_id, str(cls_id))
            cls_lower = cls_name.lower()

            # collect names for UI
            detected_signs.append(cls_name)

            # Draw bounding box + label
            color = (0, 255, 0)  # green
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls_name} {conf_box*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label_text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # ---------- RULES PER SIGN ----------

            # 1) Speed limit (works with your 10–120 limit classes)
            if cls_id in speed_limit_classes:
                limit_val = speed_limit_classes[cls_id]
                if conf_box > best_limit_conf:
                    best_limit_conf = conf_box
                    current_frame_limit = limit_val
                if vehicle_speed > limit_val + 2:
                    any_violation = True
                    violated_msgs.append(f"Over Speed Limit {int(limit_val)}")

            # 2) Red light (your class name is exactly "Red Light")
            if cls_lower == "red light":
                if car_moving:
                    any_violation = True
                    violated_msgs.append("Red Light Violation")

            # 3) Stop sign (your class name is exactly "Stop")
            if cls_lower == "stop":
                if car_moving:
                    any_violation = True
                    violated_msgs.append("Stop Sign Violation")

            # 4) No Parking (only if dataset has such label in future)
            if "no parking" in cls_lower or "no-parking" in cls_lower:
                if is_parked:
                    any_violation = True
                    violated_msgs.append("No Parking Violation")

            # 5) Turn prohibition signs (future extension)
            turn_type = classify_turn_sign(cls_lower)
            if turn_type == "no_left" and turning_left:
                any_violation = True
                violated_msgs.append("No Left Turn Violation")
            if turn_type == "no_right" and turning_right:
                any_violation = True
                violated_msgs.append("No Right Turn Violation")
            if turn_type == "no_uturn" and doing_uturn:
                any_violation = True
                violated_msgs.append("No U-Turn Violation")

    # update stored limit for HUD
    now = time.time()
    if current_frame_limit is not None:
        last_limit_value = current_frame_limit
        last_limit_time = now
    else:
        if last_limit_value is not None and now - last_limit_time > LIMIT_HOLD_SEC:
            last_limit_value = None

    # HUD
    cv2.putText(vis, f"Speed: {vehicle_speed:.1f} km/h",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    if last_limit_value is not None:
        cv2.putText(vis, f"Limit: {last_limit_value:.0f} km/h",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # Banner if any violations
    if any_violation:
        text = "SIGN VIOLATED"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, th + 40), (0, 0, 255), -1)
        vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

        cv2.putText(vis, text, (20, th + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    (255, 255, 255), 3, cv2.LINE_AA)

        y_msg = th + 50
        for msg in sorted(set(violated_msgs)):
            cv2.putText(vis, msg, (20, y_msg),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y_msg += 30

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(rgb)

    return pil_out, last_limit_value, last_limit_time, any_violation, violated_msgs, detected_signs


def infer_video_with_violations(
    model,
    video_path: str,
    class_names,
    speed_limit_classes,
    vehicle_speed: float,
    car_moving: bool,
    manoeuvre: str,
    conf: float,
    imgsz: int,
    progress_callback=None,
    preview_every=8,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    start = time.time()
    last_preview = None

    last_limit_value = None
    last_limit_time = 0.0

    sign_counter = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        (pil_frame,
         last_limit_value,
         last_limit_time,
         any_violation,
         violated_msgs,
         detected_signs) = process_frame_with_violations(
            model=model,
            frame_bgr=frame,
            class_names=class_names,
            speed_limit_classes=speed_limit_classes,
            vehicle_speed=vehicle_speed,
            car_moving=car_moving,
            manoeuvre=manoeuvre,
            last_limit_value=last_limit_value,
            last_limit_time=last_limit_time,
            conf=conf,
            imgsz=imgsz,
        )

        sign_counter.update(detected_signs)

        bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        out.write(bgr)

        if progress_callback and total > 0:
            progress_callback(frame_idx / total)

        if frame_idx % preview_every == 0:
            last_preview = pil_frame.copy()

    cap.release()
    out.release()
    elapsed = time.time() - start

    return out_path, elapsed, frame_idx, last_preview, sign_counter


# ================== STREAMLIT UI ==================

st.title("🚦 Traffic Sign Detection & Violation (YOLO + Streamlit)")
st.write(
    "• Detects each signboard with bounding box + label\n"
    "• Rules: Speed Limit, Red Light, Stop (+ optional No Parking / No Turn if trained)\n"
    "• Shows **SIGN VIOLATED** + which rule(s) when broken"
)

if not DATA_PATH.exists():
    st.error(f"data.yaml not found at: {DATA_PATH}")
    st.stop()

class_names = load_class_names(DATA_PATH)
speed_limit_classes = build_speed_limit_mapping(class_names)

with st.expander("Debug: Classes & Speed-limit mapping", expanded=False):
    st.json(class_names)
    st.write("Speed-limit mapping:", speed_limit_classes)

# Model selection
weights = find_run_weights(MODEL_SEARCH_DIR)
weight_options = [w[0] for w in weights]
weight_paths = {w[0]: w[1] for w in weights}

if weight_options:
    selected = st.selectbox("Select trained weights from runs/detect", weight_options, index=0)
    selected_model_path = weight_paths[selected]
else:
    st.warning(f"No trained weights found under {MODEL_SEARCH_DIR}. Falling back to {DEFAULT_MODEL_FILE}.")
    selected_model_path = str(DEFAULT_MODEL_FILE)

col1, col2 = st.columns([1, 2])

with col1:
    device = st.selectbox("Device", ["cpu", "cuda:0"], index=0)
    conf = st.slider("Confidence threshold", 0.1, 1.0, CONF_DEFAULT, 0.05)
    imgsz = st.selectbox("Image size", [320, 416, 512, 640, 960], index=3)
    vehicle_speed = st.slider("Vehicle speed (km/h)", 0.0, 160.0, 60.0, 1.0)
    car_moving = st.checkbox("Vehicle is moving", value=True)
    manoeuvre = st.selectbox(
        "Vehicle manoeuvre",
        [
            "Driving straight",
            "Parked / standing",
            "Turning left",
            "Turning right",
            "U-turn",
        ],
        index=0,
    )

with col2:
    uploaded = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","bmp","mp4","mov","avi","mkv"])

# Load model
model = None
if selected_model_path:
    try:
        with st.spinner(f"Loading model: {selected_model_path}"):
            model = load_model_safe(selected_model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

# Handle upload
if uploaded and model:
    name = uploaded.name
    ext = Path(name).suffix.lower()
    file_bytes = uploaded.read()

    # IMAGE
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        st.info(f"Running image inference on: {name}")
        try:
            pil_in = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_in), cv2.COLOR_RGB2BGR)

            (pil_out,
             _,
             _,
             any_violation,
             violated_msgs,
             detected_signs) = process_frame_with_violations(
                model=model,
                frame_bgr=bgr,
                class_names=class_names,
                speed_limit_classes=speed_limit_classes,
                vehicle_speed=vehicle_speed,
                car_moving=car_moving,
                manoeuvre=manoeuvre,
                last_limit_value=None,
                last_limit_time=0.0,
                conf=conf,
                imgsz=imgsz,
            )

            st.image(pil_out, caption="Annotated image", use_column_width=True)

            if detected_signs:
                unique_signs = sorted(set(detected_signs))
                st.info("Sign boards detected: " + ", ".join(unique_signs))
            else:
                st.info("No sign boards detected.")

            if any_violation:
                st.error("SIGN VIOLATED: " + ", ".join(sorted(set(violated_msgs))))
            else:
                st.success("No sign violations detected in this image.")

            buf = io.BytesIO()
            pil_out.save(buf, format="JPEG")
            st.download_button(
                "Download annotated image",
                data=buf.getvalue(),
                file_name=Path(name).stem + "_violations.jpg",
                mime="image/jpeg",
            )

        except Exception as e:
            st.error(f"Image inference failed: {e}")

    # VIDEO
    elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
        st.info(f"Running video inference on: {name}")
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        video_path = tmp.name

        try:
            progress_bar = st.progress(0.0)
            status = st.empty()
            preview_slot = st.empty()

            def progress_cb(ratio):
                progress_bar.progress(min(max(ratio, 0.0), 1.0))

            (out_path,
             elapsed,
             frames,
             preview_img,
             sign_counter) = infer_video_with_violations(
                model=model,
                video_path=video_path,
                class_names=class_names,
                speed_limit_classes=speed_limit_classes,
                vehicle_speed=vehicle_speed,
                car_moving=car_moving,
                manoeuvre=manoeuvre,
                conf=conf,
                imgsz=imgsz,
                progress_callback=progress_cb,
            )

            progress_bar.progress(1.0)
            status.success(f"Done — {frames} frames, {elapsed:.1f}s. Saved: {out_path}")

            if preview_img:
                preview_slot.image(preview_img, caption="Preview (sample frame)", use_column_width=True)

            # Show sign statistics
            if sign_counter:
                st.info("Sign boards detected in this video (counts):")
                st.json(dict(sign_counter))
            else:
                st.info("No sign boards detected in this video.")

            with open(out_path, "rb") as f:
                video_data = f.read()
                st.video(video_data)
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    f.read(),
                    file_name=Path(name).stem + "_violations.mp4",
                    mime="video/mp4",
                )

        except Exception as e:
            st.error(f"Video inference failed: {e}")
        finally:
            try:
                os.remove(video_path)
            except Exception:
                pass

    else:
        st.warning("Unsupported file type. Upload image or video.")

st.markdown("---")
st.caption("Uses your custom YOLO model (Green/Red Light, Speed Limits, Stop) + rule-based violation logic.")
