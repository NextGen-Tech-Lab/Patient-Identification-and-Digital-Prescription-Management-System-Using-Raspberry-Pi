"""
enroll_patient.py
-----------------
Interactive command-line application to manually enroll a new patient
into the Smart Medicine Dispenser database.

The form collects:
  - Patient ID (auto-generated or custom)
  - Demographics: gender, age
  - Prescription: drug name, dose, unit, route, doses per day
  - Face image: capture from webcam OR provide existing file path

Saves to:
  - data/metadata/patient_database.json    (updated)
  - data/metadata/patient_face_map.csv     (updated)
  - data/raw/{patient_id}/{patient_id}_base.jpg  (image saved here)
"""

import json
import csv
import os
import sys
import shutil
import time
import datetime
import threading
import numpy as np
import cv2
from pathlib import Path

try:
    from hardware_io import hw
except ImportError:
    try:
        from scripts.hardware_io import hw
    except ImportError:
        hw = None

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
DB_PATH = METADATA_DIR / "patient_database.json"
MAP_CSV_PATH = METADATA_DIR / "patient_face_map.csv"

# ──────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print("=" * 60)
    print("  SMART MEDICINE DISPENSER — Patient Enrollment Form")
    print("=" * 60)
    print()

def prompt(question, default=None, required=True):
    """Prompt the user for input with optional default."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"  {question}{hint}: ").strip()
        if val == "" and default is not None:
            return str(default)
        if val == "" and required:
            print("  [!] This field is required. Please enter a value.")
            continue
        if val:
            return val

def prompt_choice(question, choices):
    """Prompt for a choice from a list, allowing selection by name or number."""
    choices_display = " / ".join([f"{i+1}:{c}" for i, c in enumerate(choices)])
    while True:
        val = input(f"  {question} ({choices_display}): ").strip().upper()
        
        # Support numeric selection (1, 2, 3...)
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return choices[idx].upper()
                
        # Support word selection (AUTO, CUSTOM, etc)
        if val in [c.upper() for c in choices]:
            return val
            
        print(f"  [!] Please enter a valid number (1-{len(choices)}) or name.")

def prompt_number(question, min_val=0, max_val=999, default=None):
    """Prompt for a numeric value within a range."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"  {question}{hint}: ").strip()
        if val == "" and default is not None:
            return default
        try:
            num = float(val)
            if min_val <= num <= max_val:
                return num
            print(f"  [!] Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  [!] Please enter a valid number.")

def load_database():
    """Load the existing patient database, or return empty dict if not found."""
    if DB_PATH.exists():
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_database(db):
    """Save the patient database to JSON."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def append_to_map_csv(subject_id, original_path, mapped_path):
    """Append a new row to the patient_face_map.csv."""
    MAP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = MAP_CSV_PATH.exists()
    with open(MAP_CSV_PATH, "a", newline="") as csvfile:
        fieldnames = ["subject_id", "original_image_path", "mapped_image_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "subject_id": subject_id,
            "original_image_path": str(original_path),
            "mapped_image_path": mapped_path
        })

def generate_patient_id(db):
    """Auto-generate a new unique patient ID like MAN001, MAN002..."""
    existing_manual = [k for k in db.keys() if k.startswith("MAN")]
    if not existing_manual:
        return "MAN001"
    nums = [int(k[3:]) for k in existing_manual if k[3:].isdigit()]
    return f"MAN{(max(nums) + 1):03d}"

# FACE CAPTURE — 12-frame auto-scan
# ──────────────────────────────────────────────────────────────

# Shared CLAHE instance (same as recognize_patient.py for consistency)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class VideoStream:
    """Threaded video stream for zero-latency capture."""
    def __init__(self, src):
        self.src = src
        if os.name == 'nt' and isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        for _ in range(10):
            if self.cap.isOpened():
                ret, f = self.cap.read()
                if ret:
                    self.ret = True
                    self.frame = f
                    break
            time.sleep(0.1)

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

def _open_cam_for_enrollment():
    """Open webcam based on user input to support multiple cameras."""
    while True:
        idx_str = input("\n  Enter camera index (0, 1) or IP stream URL (http://...) or 'q' to quit [0]: ").strip()
        
        if idx_str.lower() == 'q':
            print("  [QUIT] Exiting...")
            sys.exit(0)
            
        if idx_str.startswith("http://") or idx_str.startswith("https://") or idx_str.startswith("rtsp://"):
            idx = idx_str
        else:
            idx = int(idx_str) if idx_str.isdigit() else 0

        print(f"  [CAM] Opening camera source: {idx} ...")
        stream = VideoStream(idx)
        if stream.isOpened():
            print(f"  [CAM] Camera ready at: {idx}.")
            return stream
        else:
            print(f"  [!] Could not open camera at: {idx}. Please try another.")

def _make_detector():
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(cascade)
    return None if det.empty() else det

def capture_webcam_frames(patient_id, n_frames=12):
    """
    Auto-capture n_frames face images from webcam with 480p preview canvas.
    """
    save_dir = RAW_DIR / patient_id
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print()
    print(f"  Opening camera — scanning face for patient {patient_id}")
    print(f"  Look directly at the camera. {n_frames} frames will be captured automatically.")
    print("  Press [Q] to cancel.")
    print()

    stream = _open_cam_for_enrollment()
    if stream is None:
        print("  [!] Cannot open webcam.")
        return 0, None

    detector    = _make_detector()
    saved       = 0
    attempts    = 0
    last_save   = 0
    first_path  = None
    win_name    = f"Face Enrollment — {patient_id}"

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 480, 320)

    while saved < n_frames and attempts < 300:
        ret, frame = stream.read()
        if not ret or frame is None:
            attempts += 1
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        attempts += 1
        face_rect = None

        if detector:
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            if len(faces) > 0:
                face_rect = max(faces, key=lambda f: f[2] * f[3])

        # 480x320 Canvas Render
        canvas = np.full((320, 480, 3), (20, 16, 12), dtype=np.uint8)
        
        # Header
        cv2.rectangle(canvas, (0, 0), (480, 30), (32, 24, 18), -1)
        cv2.putText(canvas, f"+ ENROLLING PATIENT: {patient_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{saved}/{n_frames}", (410, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (50, 200, 100), 1, cv2.LINE_AA)

        # Scale and fit preview
        cam_disp = cv2.resize(frame, (320, 210))
        if face_rect is not None:
            fx, fy, fw, fh = face_rect
            sx, sy = 320.0 / frame.shape[1], 210.0 / frame.shape[0]
            fx_s, fy_s, fw_s, fh_s = int(fx * sx), int(fy * sy), int(fw * sx), int(fh * sy)
            cv2.rectangle(cam_disp, (fx_s, fy_s), (fx_s+fw_s, fy_s+fh_s), (50, 210, 80), 2)

        canvas[36:246, 80:400] = cam_disp
        cv2.rectangle(canvas, (80, 36), (400, 246), (70, 55, 45), 1)

        # Progress bar
        bar_w = 400 - 80
        prog_w = int((saved / n_frames) * bar_w)
        cv2.rectangle(canvas, (80, 254), (80 + bar_w, 268), (35, 26, 20), -1)
        cv2.rectangle(canvas, (80, 254), (80 + prog_w, 268), (50, 190, 40), -1)
        cv2.rectangle(canvas, (80, 254), (80 + bar_w, 268), (70, 55, 45), 1)

        # Bottom Bar
        cv2.rectangle(canvas, (0, 284), (480, 320), (16, 12, 10), -1)
        cv2.putText(canvas, "[Q] CANCEL ENROLLMENT", (160, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 220), 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("  Capture cancelled.")
            break

        # Save frame every 0.4 s when face is detected
        now = time.time()
        if face_rect is not None and (now - last_save) >= 0.4:
            save_path = save_dir / f"webcam_{ts}_{saved:03d}.jpg"
            cv2.imwrite(str(save_path), frame)
            if first_path is None:
                first_path = save_path
            saved += 1
            last_save = now
            print(f"    Frame {saved}/{n_frames} captured")
            if hw:
                hw.chirp()

    stream.release()
    cv2.destroyAllWindows()

    if saved == n_frames:
        print(f"  [OK] {saved} frames saved for {patient_id}.")
        if hw:
            hw.enroll_success()
    else:
        print(f"  [!] Only {saved} of {n_frames} frames captured.")
    return saved, first_path

def copy_existing_image(source_path_str, dest_dir, patient_id):
    """Copy a user-provided image file to the patient's directory."""
    source = Path(source_path_str.strip('"').strip("'"))
    if not source.exists():
        print(f"  [!] File not found: {source}")
        return None
    if source.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        print("  [!] Unsupported format. Use JPG, PNG or BMP.")
        return None
    dest = dest_dir / f"{patient_id}_base{source.suffix.lower()}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"  [OK] Image copied to: {dest}")
    return dest

# ──────────────────────────────────────────────────────────────
# FORM SECTIONS
# ──────────────────────────────────────────────────────────────

def collect_patient_id(db):
    print("  STEP 1 of 4 — Patient Identification")
    print("  " + "-" * 40)
    auto_id = generate_patient_id(db)
    print(f"  Auto-generated Patient ID: {auto_id}")
    choice = prompt_choice("  Use auto-generated ID or enter a custom one?", ["AUTO", "CUSTOM"])

    if choice == "AUTO":
        patient_id = auto_id
    else:
        while True:
            custom = prompt("  Enter custom Patient ID (e.g. PAT999)", required=True)
            if custom in db:
                print(f"  ✗ Patient ID '{custom}' already exists. Choose another.")
            else:
                patient_id = custom
                break

    print(f"  ✓ Patient ID: {patient_id}")
    return patient_id

def collect_demographics():
    print()
    print("  STEP 2 of 4 — Patient Demographics")
    print("  " + "-" * 40)

    gender = prompt_choice("  Gender", ["M", "F", "O"])
    age = int(prompt_number("  Age (years)", min_val=0, max_val=120))

    return {"gender": gender, "anchor_age": age}

def collect_prescription():
    print()
    print("  STEP 3 of 4 — Prescription Details")
    print("  " + "-" * 40)
    print("  (This determines what medicine the dispenser will give)")
    print()

    drug = prompt("  Medicine / Drug name (e.g. Paracetamol, Insulin)")
    dose_val = prompt("  Dose value (e.g. 500, 10, 0.5)")
    dose_unit = prompt("  Dose unit (e.g. mg, mcg, UNIT, mL)")

    print()
    print("  Common routes: PO (oral), IV (intravenous), SC (subcutaneous),")
    print("                 IM (intramuscular), TOP (topical), INH (inhaled)")
    route = prompt("  Administration route", default="PO")

    doses_per_day = int(prompt_number("  Doses per 24 hours (e.g. 1, 2, 3, 4)", min_val=1, max_val=24, default=1))
    dispense_seconds = int(prompt_number("  Dispenser active duration in seconds (e.g. 2, 3, 5)", min_val=1, max_val=30, default=3))

    return {
        "drug": drug,
        "dose_val": dose_val,
        "dose_unit": dose_unit,
        "route": route.upper(),
        "doses_per_24_hrs": str(doses_per_day),
        "dispense_duration_seconds": dispense_seconds
    }

def collect_face_image(patient_id):
    print()
    print("  STEP 4 of 4 — Face Scan")
    print("  " + "-" * 40)
    print("  The system will auto-capture 12 frames of your face from the webcam.")
    print("  These frames are used to train the face recognition model.")
    print()

    save_dir = RAW_DIR / patient_id
    first_image_path = None
    original_ref = "webcam_scan"

    while True:
        method = prompt_choice(
            "  Capture method",
            ["WEBCAM", "FILE"]
        )

        if method == "WEBCAM":
            saved, first_image_path = capture_webcam_frames(patient_id, n_frames=12)
            if saved >= 6:
                # Enough frames captured — use the first one as the base image ref
                rel_path = f"data/raw/{patient_id}/" + first_image_path.name
                return rel_path, "webcam_scan"
            elif saved > 0:
                print(f"  [!] Only {saved} frames captured (need at least 6). Try again.")
            else:
                print("  [!] No frames captured. Choose FILE to provide an image instead.")

        elif method == "FILE":
            file_path = prompt("  Enter the full path to the image file")
            dest = copy_existing_image(file_path, save_dir, patient_id)
            if dest:
                rel_path = f"data/raw/{patient_id}/{dest.name}"
                return rel_path, file_path

# ──────────────────────────────────────────────────────────────
# SUMMARY & CONFIRMATION
# ──────────────────────────────────────────────────────────────

def show_summary(patient_id, demographics, prescription, image_path):
    print()
    print("=" * 60)
    print("  ENROLLMENT SUMMARY — Please Confirm")
    print("=" * 60)
    print(f"  Patient ID      : {patient_id}")
    print(f"  Gender          : {demographics['gender']}")
    print(f"  Age             : {demographics['anchor_age']} years")
    print()
    print(f"  Drug            : {prescription['drug']}")
    print(f"  Dose            : {prescription['dose_val']} {prescription['dose_unit']}")
    print(f"  Route           : {prescription['route']}")
    print(f"  Doses / 24h     : {prescription['doses_per_24_hrs']}")
    print(f"  Dispense Time   : {prescription['dispense_duration_seconds']}s")
    print()
    print(f"  Face Image Path : {image_path}")
    print("=" * 60)
    confirm = prompt_choice("  Save this patient to the database?", ["YES", "NO"])
    return confirm == "YES"

# ──────────────────────────────────────────────────────────────
# MAIN ENROLLMENT FLOW
# ──────────────────────────────────────────────────────────────

def enroll_patient():
    clear()
    banner()

    db = load_database()
    print(f"  Current database: {len(db)} enrolled patient(s).")
    print()

    # Collect all information
    patient_id = collect_patient_id(db)
    demographics = collect_demographics()
    prescription = collect_prescription()
    image_rel_path, original_path = collect_face_image(patient_id)

    # Show summary and confirm
    confirmed = show_summary(patient_id, demographics, prescription, image_rel_path)

    if not confirmed:
        print()
        print("  Enrollment cancelled. No changes saved.")
        return

    # Build patient record
    patient_record = {
        "subject_id": patient_id,
        "gender": demographics["gender"],
        "anchor_age": demographics["anchor_age"],
        "prescription": prescription,
        "enrolled": True,
        "enrolled_date": datetime.date.today().isoformat(),
        "face_image_path": image_rel_path,
        "enrollment_source": "manual"
    }

    # Save to database
    db[patient_id] = patient_record
    save_database(db)

    # Append to mapping CSV
    append_to_map_csv(patient_id, original_path, image_rel_path)

    print()
    print("=" * 60)
    print(f"  [OK] Patient {patient_id} successfully enrolled!")
    print(f"  [OK] Database now contains {len(db)} patient(s).")
    print("=" * 60)
    print()

def main():
    while True:
        enroll_patient()
        print()
        again = prompt_choice("  Enroll another patient?", ["YES", "NO"])
        if again == "NO":
            print()
            print("  Exiting enrollment form. Goodbye!")
            print()
            break
        clear()
        banner()

if __name__ == "__main__":
    main()
