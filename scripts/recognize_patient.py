"""
recognize_patient.py
--------------------
Real-Time Face Recognition + Medical Dispenser Kiosk HUD.
Optimized for Raspberry Pi 480p Displays (480x320) & Zero-Latency Video Streaming.

KEYBOARD CONTROLS (in live window):
  [C]   -> Capture face frames to enroll / update patient
  [E]   -> Edit patient prescription (in-terminal editor)
  [K]   -> Lock / Unlock currently displayed patient
  [SPC] -> Freeze / Unfreeze camera preview
  [F]   -> Toggle Fullscreen mode (hides Raspberry Pi desktop bar)
  [R]   -> Force re-train all SFace embeddings
  [Q]   -> Quit dispenser system
"""

import cv2
import json
import time
import os
import sys
import threading
import datetime
import pickle
from pathlib import Path
import numpy as np

try:
    from hardware_io import hw
except ImportError:
    try:
        from scripts.hardware_io import hw
    except ImportError:
        hw = None

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR         = PROJECT_ROOT / "data" / "raw"
DB_PATH         = PROJECT_ROOT / "data" / "metadata" / "patient_database.json"
MODEL_DIR       = PROJECT_ROOT / "models" / "dnn"
YUNET_PATH      = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH      = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.pkl"

# ── Thresholds (SFace Cosine Similarity: 0.0 to 1.0) ──────────────────────────
MATCH_THRESHOLD   = 0.450   # Required for identification
CAUTION_THRESHOLD = 0.550   # 0.450–0.550 = LOW CONFIDENCE, >= 0.550 = AUTHORIZED

# ── Native Display Canvas (480x320 for 3.5" RPi Touchscreen / 480p display) ───
CANVAS_W = 480
CANVAS_H = 320
WINDOW_NAME = "Smart Dispenser — Face Recognition"

# ── Color Palette (Medical Dark Theme, BGR) ───────────────────────────────────
BG_CANVAS    = (20, 16, 12)     # Deep Charcoal Slate
BG_HEADER    = (32, 24, 18)     # Header Slate
BG_CARD      = (30, 22, 16)     # Card Navy Dark
BORDER_COLOR = (70, 55, 45)     # Subtle Card Outline
C_WHITE      = (255, 255, 255)  # Crisp White
C_GRAY       = (180, 180, 180)  # Dim Gray
C_CYAN       = (220, 180, 60)   # Medical Cyan Accent
C_GREEN      = (50, 190, 40)    # Emerald Success
C_YELLOW     = (20, 180, 240)   # Caution Gold
C_RED        = (40, 40, 210)    # Alert Red
C_ORANGE     = (30, 110, 230)   # Locked Amber


# ── Threaded Zero-Latency Video Stream ────────────────────────────────────────
class VideoStream:
    """
    Lightweight background reader thread that constantly consumes camera frames.
    Completely eliminates FFmpeg/OpenCV network frame buffering and 10-second lag!
    """
    def __init__(self, src):
        self.src = src
        print(f"[CAM] Initializing stream from: {src} ...")
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

        # Warm-up / initial frame read
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


# ── Deep Learning Models (OpenCV Native YuNet & SFace) ────────────────────────
def make_detector(size=(640, 480)):
    if not YUNET_PATH.exists():
        print(f"[ERROR] YuNet model missing: {YUNET_PATH}")
        sys.exit(1)
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(
            model=str(YUNET_PATH), config="", input_size=size,
            score_threshold=0.4, nms_threshold=0.3, top_k=5000
        )
    elif hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        return cv2.FaceDetectorYN.create(
            model=str(YUNET_PATH), config="", input_size=size,
            score_threshold=0.4, nms_threshold=0.3, top_k=5000
        )
    else:
        print("[ERROR] OpenCV installation does not support FaceDetectorYN.")
        sys.exit(1)


def make_recognizer():
    if not SFACE_PATH.exists():
        print(f"[ERROR] SFace model missing: {SFACE_PATH}")
        sys.exit(1)
    if hasattr(cv2, "FaceRecognizerSF_create"):
        return cv2.FaceRecognizerSF_create(model=str(SFACE_PATH), config="")
    elif hasattr(cv2, "FaceRecognizerSF") and hasattr(cv2.FaceRecognizerSF, "create"):
        return cv2.FaceRecognizerSF.create(model=str(SFACE_PATH), config="")
    else:
        print("[ERROR] OpenCV installation does not support FaceRecognizerSF.")
        sys.exit(1)


def detect_largest_face(img_bgr, detector):
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    ret, faces = detector.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


# ── SFace Embeddings Extraction ───────────────────────────────────────────────
def train(db):
    print("[TRAIN] Building SFace embeddings ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    detector   = make_detector()
    recognizer = make_recognizer()
    embeddings = {}
    skipped = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for subject_id, patient in db.items():
        patient_dir = RAW_DIR / subject_id
        if not patient_dir.exists():
            skipped += 1
            continue

        images = [p for p in patient_dir.iterdir() if p.suffix.lower() in exts]
        subject_feats = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            h, w = img.shape[:2]
            scale = 1.0
            if w > 800:
                scale = 800 / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            face = detect_largest_face(img, detector)
            if face is None: continue
                
            aligned = recognizer.alignCrop(img, face)
            feat = recognizer.feature(aligned)
            subject_feats.append(feat[0])

        if subject_feats:
            mean_feat = np.mean(subject_feats, axis=0).reshape(1, 128)
            embeddings[subject_id] = mean_feat
        else:
            skipped += 1

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"[TRAIN] Done: Extracted embeddings for {len(embeddings)} patients ({skipped} skipped).")
    return embeddings, {}


def load_or_train(db):
    if EMBEDDINGS_PATH.exists():
        print("[MODEL] Loading cached SFace embeddings ...")
        with open(EMBEDDINGS_PATH, "rb") as f:
            emb = pickle.load(f)
        print(f"[MODEL] {len(emb)} patients enrolled.")
        return emb, {}
    return train(db)


def load_db():
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)
    with open(DB_PATH) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {p["subject_id"]: p for p in data if isinstance(p, dict) and "subject_id" in p}
    return {}


def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)


# ── Webcam capture enrollment ─────────────────────────────────────────────────
def capture_webcam_frames(stream, detector, subject_id, n_frames=10):
    save_dir = RAW_DIR / subject_id
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    saved = 0
    attempts = 0
    last_capture = 0
    print(f"\n[CAPTURE] Starting capture for {subject_id}. Look at camera ...")

    while saved < n_frames and attempts < 150:
        ret, frame = stream.read()
        if not ret or frame is None:
            attempts += 1
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        rect = detect_largest_face(frame, detector)
        attempts += 1

        # 480p preview canvas
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_CANVAS, dtype=np.uint8)
        cv2.putText(canvas, f"ENROLLING: {subject_id}", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Saved: {saved}/{n_frames}", (CANVAS_W - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

        # Scale preview
        cam_disp = cv2.resize(frame, (320, 240))
        canvas[40:280, 80:400] = cam_disp

        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

        now = time.time()
        if rect is not None and (now - last_capture) > 0.4:
            save_path = save_dir / f"webcam_{ts}_{saved:03d}.jpg"
            cv2.imwrite(str(save_path), frame)
            saved += 1
            last_capture = now
            print(f"  Saved frame {saved}/{n_frames}: {save_path.name}")

    print(f"[CAPTURE] Finished: {saved} frame(s) saved for {subject_id}.\n")
    return saved


# ── 480p Flawless UI Renderer ─────────────────────────────────────────────────
def render_480p_dashboard(frame, rect, patient, dist, status, locked, is_frozen, eval_countdown):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_CANVAS, dtype=np.uint8)

    # 1. Header Bar (y: 0..30)
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 30), BG_HEADER, -1)
    cv2.line(canvas, (0, 30), (CANVAS_W, 30), BORDER_COLOR, 1)
    cv2.putText(canvas, "+ SMART MED DISPENSER", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 1, cv2.LINE_AA)

    # Status Pill
    if locked:
        pill_txt, pill_col, txt_col = "LOCKED", C_ORANGE, (0, 0, 0)
    elif status == "AUTHORIZED":
        pill_txt, pill_col, txt_col = "AUTHORIZED", C_GREEN, (0, 0, 0)
    elif status == "LOW_CONFIDENCE":
        pill_txt, pill_col, txt_col = "CAUTION", C_YELLOW, (0, 0, 0)
    else:
        pill_txt, pill_col, txt_col = "SCANNING", (60, 50, 45), C_WHITE

    pill_w = 115
    pill_x = CANVAS_W - pill_w - 8
    cv2.rectangle(canvas, (pill_x, 5), (pill_x + pill_w, 25), pill_col, -1)
    cv2.putText(canvas, pill_txt, (pill_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_col, 1, cv2.LINE_AA)

    # 2. Left Camera Box (x: 8, y: 34, w: 220, h: 244)
    cam_x, cam_y, cam_w, cam_h = 8, 34, 220, 244
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), (25, 20, 16), -1)
    cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), BORDER_COLOR, 1)

    if frame is not None:
        # Center-crop & fit frame into 220x244
        H, W = frame.shape[:2]
        scale = max(cam_w / W, cam_h / H)
        resized = cv2.resize(frame, (int(W * scale), int(H * scale)))
        cy = max(0, (resized.shape[0] - cam_h) // 2)
        cx = max(0, (resized.shape[1] - cam_w) // 2)
        disp_cam = resized[cy:cy+cam_h, cx:cx+cam_w].copy()

        # Map face coordinates if detected
        if rect is not None:
            fx, fy, fw, fh = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            fx_d = max(0, min(cam_w - 1, int(fx * scale) - cx))
            fy_d = max(0, min(cam_h - 1, int(fy * scale) - cy))
            fw_d = min(cam_w - fx_d, int(fw * scale))
            fh_d = min(cam_h - fy_d, int(fh * scale))

            bcol = C_ORANGE if locked else (C_GREEN if status == "AUTHORIZED" else (C_YELLOW if status == "LOW_CONFIDENCE" else C_CYAN))
            # Sleek Corner Brackets
            c_len = min(20, fw_d // 3)
            # Top-left
            cv2.line(disp_cam, (fx_d, fy_d), (fx_d + c_len, fy_d), bcol, 2)
            cv2.line(disp_cam, (fx_d, fy_d), (fx_d, fy_d + c_len), bcol, 2)
            # Top-right
            cv2.line(disp_cam, (fx_d + fw_d, fy_d), (fx_d + fw_d - c_len, fy_d), bcol, 2)
            cv2.line(disp_cam, (fx_d + fw_d, fy_d), (fx_d + fw_d, fy_d + c_len), bcol, 2)
            # Bottom-left
            cv2.line(disp_cam, (fx_d, fy_d + fh_d), (fx_d + c_len, fy_d + fh_d), bcol, 2)
            cv2.line(disp_cam, (fx_d, fy_d + fh_d), (fx_d, fy_d + fh_d - c_len), bcol, 2)
            # Bottom-right
            cv2.line(disp_cam, (fx_d + fw_d, fy_d + fh_d), (fx_d + fw_d - c_len, fy_d + fh_d), bcol, 2)
            cv2.line(disp_cam, (fx_d + fw_d, fy_d + fh_d), (fx_d + fw_d, fy_d + fh_d - c_len), bcol, 2)

        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = disp_cam

    # Live / 1s Scan tag overlay
    tag_txt = "FROZEN" if is_frozen else f"* LIVE ({eval_countdown:.1f}s)"
    tag_col = C_YELLOW if is_frozen else C_GREEN
    cv2.rectangle(canvas, (cam_x + 6, cam_y + 6), (cam_x + 92, cam_y + 22), (20, 16, 12), -1)
    cv2.putText(canvas, tag_txt, (cam_x + 10, cam_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tag_col, 1, cv2.LINE_AA)

    # 3. Right Patient & Prescription HUD Box (x: 236, y: 34, w: 236, h: 244)
    hud_x, hud_y, hud_w, hud_h = 236, 34, 236, 244
    cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), BG_CARD, -1)
    cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), BORDER_COLOR, 1)

    if patient:
        # Patient Thumbnail
        thumb_w, thumb_h = 56, 56
        thumb_x, thumb_y = hud_x + hud_w - thumb_w - 8, hud_y + 8
        img_rel = patient.get("face_image_path", "")
        img_loaded = False
        if img_rel:
            img_path = PROJECT_ROOT / img_rel
            if img_path.exists():
                try:
                    face_img = cv2.imread(str(img_path))
                    if face_img is not None:
                        face_img = cv2.resize(face_img, (thumb_w, thumb_h))
                        canvas[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w] = face_img
                        img_loaded = True
                except Exception:
                    pass
        if not img_loaded:
            cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x+thumb_w, thumb_y+thumb_h), (50, 40, 30), -1)
            cv2.putText(canvas, "PHOTO", (thumb_x + 8, thumb_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1)
        cv2.rectangle(canvas, (thumb_x, thumb_y), (thumb_x+thumb_w, thumb_y+thumb_h), C_CYAN, 1)

        # Patient Demographics
        pid = patient.get("subject_id", "?")
        gender = patient.get("gender", "?")
        age = patient.get("anchor_age", "?")
        cv2.putText(canvas, f"ID: {pid}", (hud_x + 10, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{gender}  |  {age} yrs", (hud_x + 10, hud_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)

        # Match Similarity Bar
        conf_pct = max(0.0, min(100.0, dist * 100.0))
        cv2.putText(canvas, f"Match: {conf_pct:.1f}%", (hud_x + 10, hud_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREEN if dist >= MATCH_THRESHOLD else C_YELLOW, 1, cv2.LINE_AA)
        bar_x, bar_y, bar_w, bar_h = hud_x + 10, hud_y + 70, 110, 6
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 40, 30), -1)
        fill_w = int(bar_w * (conf_pct / 100.0))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), C_GREEN if dist >= MATCH_THRESHOLD else C_YELLOW, -1)

        # Divider
        cv2.line(canvas, (hud_x + 8, hud_y + 84), (hud_x + hud_w - 8, hud_y + 84), BORDER_COLOR, 1)

        # Prescription Details Card
        prescriptions = patient.get("prescriptions", [])
        if not prescriptions and "prescription" in patient:
            prescriptions = [patient["prescription"]]
        rx = prescriptions[0] if prescriptions else {}

        rx_box_y = hud_y + 90
        cv2.putText(canvas, "Rx PRESCRIPTION", (hud_x + 10, rx_box_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_CYAN, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (hud_x + 8, rx_box_y + 22), (hud_x + hud_w - 8, hud_y + hud_h - 10), (22, 17, 13), -1)
        cv2.rectangle(canvas, (hud_x + 8, rx_box_y + 22), (hud_x + hud_w - 8, hud_y + hud_h - 10), BORDER_COLOR, 1)

        drug_name = rx.get("drug", "None")
        if len(drug_name) > 20: drug_name = drug_name[:18] + ".."
        dose_val  = rx.get("dose_val", "")
        dose_unit = rx.get("dose_unit", "")
        route     = rx.get("route", "PO")
        freq      = rx.get("doses_per_24_hrs", "1")

        dy_start = rx_box_y + 40
        cv2.putText(canvas, f"Drug: {drug_name}", (hud_x + 14, dy_start), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Dose: {dose_val} {dose_unit}", (hud_x + 14, dy_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Route: {route}", (hud_x + 14, dy_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Freq: {freq}x / 24 hrs", (hud_x + 14, dy_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)
        
        status_disp = "DISPENSE READY" if status == "AUTHORIZED" else ("PATIENT PINNED" if locked else "VERIFY IDENTITY")
        disp_col    = C_GREEN if status == "AUTHORIZED" else (C_ORANGE if locked else C_YELLOW)
        cv2.putText(canvas, f"+ {status_disp}", (hud_x + 14, dy_start + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.36, disp_col, 1, cv2.LINE_AA)
    else:
        # Empty / Scanning State
        cv2.putText(canvas, "PATIENT SCANNER", (hud_x + 14, hud_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Look directly at camera", (hud_x + 14, hud_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Evaluating once/second", (hud_x + 14, hud_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1, cv2.LINE_AA)
        
        cv2.line(canvas, (hud_x + 8, hud_y + 110), (hud_x + hud_w - 8, hud_y + 110), BORDER_COLOR, 1)
        cv2.putText(canvas, "DISPENSER STATUS:", (hud_x + 14, hud_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "LOCKED - STANDBY", (hud_x + 14, hud_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Dispensing is locked until", (hud_x + 14, hud_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, "face match >= 0.450", (hud_x + 14, hud_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_GRAY, 1, cv2.LINE_AA)

    # 4. Bottom Control Bar (Always 100% visible, high contrast)
    dock_y = 284
    cv2.rectangle(canvas, (0, dock_y), (CANVAS_W, CANVAS_H), (16, 12, 10), -1)
    cv2.line(canvas, (0, dock_y), (CANVAS_W, dock_y), BORDER_COLOR, 1)

    lock_btn_txt = "[K] UNLOCK" if locked else "[K] LOCK"
    buttons = [
        ("[C] CAP", (180, 120, 20)),
        ("[E] EDIT", (20, 170, 240)),
        (lock_btn_txt, (30, 100, 230)),
        ("[SPC] FREEZ", (160, 60, 130)),
        ("[F] FULL", (130, 130, 40)),
        ("[Q] QUIT", (40, 40, 200)),
    ]

    btn_w = 72
    pad = (CANVAS_W - (len(buttons) * btn_w)) // (len(buttons) + 1)
    bx = pad
    for label, col in buttons:
        cv2.rectangle(canvas, (bx, dock_y + 5), (bx + btn_w, dock_y + 29), col, -1)
        cv2.putText(canvas, label, (bx + 5, dock_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_WHITE, 1, cv2.LINE_AA)
        bx += btn_w + pad

    return canvas


# ── Interactive Terminal Prescription Editor ──────────────────────────────────
def edit_prescription_cli(subject_id, db):
    p = db.get(subject_id)
    if not p:
        print(f"[EDIT] Patient '{subject_id}' not found.")
        return db

    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("  SMART MEDICINE DISPENSER — Patient Prescription Editor")
    print("=" * 60)
    print(f"  Patient ID : {subject_id}")
    print(f"  Gender/Age : {p.get('gender', '?')} | {p.get('anchor_age', '?')} yrs")
    print("-" * 60)

    prescriptions = p.get("prescriptions", [])
    if not prescriptions and "prescription" in p:
        prescriptions = [p["prescription"]]

    print("  Current Medicines:")
    for i, rx in enumerate(prescriptions):
        print(f"    [{i+1}] {rx.get('drug', 'Unknown')} — {rx.get('dose_val', '')} {rx.get('dose_unit', '')} ({rx.get('route', 'PO')}, {rx.get('doses_per_24_hrs', '1')}x/day)")
    if not prescriptions:
        print("    (No medicines assigned)")
    print("-" * 60)

    def ask(q, cur):
        v = input(f"  {q} [{cur}]: ").strip()
        return v if v else cur

    print("  Actions:")
    print("    1: Edit an existing medicine")
    print("    2: Add a new medicine")
    print("    3: Delete a medicine")
    print("    0: Save & Return to Camera")
    choice = input("\n  Select Option [0]: ").strip()

    if choice == '1' and prescriptions:
        try:
            idx = int(input(f"  Enter medicine index to edit (1-{len(prescriptions)}): ").strip()) - 1
            if 0 <= idx < len(prescriptions):
                rx = prescriptions[idx]
                print(f"\n  [EDITING] {rx.get('drug')}:")
                rx["drug"]             = ask("Drug Name",             rx.get("drug", ""))
                rx["dose_val"]         = ask("Dose Value",            rx.get("dose_val", ""))
                rx["dose_unit"]        = ask("Dose Unit (mL, mg, etc)",rx.get("dose_unit", ""))
                rx["route"]            = ask("Route (PO, IV, etc)",   rx.get("route", "PO"))
                rx["doses_per_24_hrs"] = ask("Doses per 24 hrs",      rx.get("doses_per_24_hrs", "1"))
                prescriptions[idx]     = rx
                p["prescriptions"]     = prescriptions
                p["last_edited"]       = datetime.datetime.now().isoformat()
                db[subject_id]         = p
                save_db(db)
                print("  ✓ Saved successfully!")
        except Exception as e:
            print(f"  [!] Error: {e}")

    elif choice == '2':
        print("\n  [ADDING NEW MEDICINE]:")
        rx = {}
        rx["drug"]             = ask("Drug Name",             "New Medicine")
        rx["dose_val"]         = ask("Dose Value",            "10")
        rx["dose_unit"]        = ask("Dose Unit (mL, mg)",    "mg")
        rx["route"]            = ask("Route (PO, IV)",        "PO")
        rx["doses_per_24_hrs"] = ask("Doses per 24 hrs",      "1")
        prescriptions.append(rx)
        p["prescriptions"] = prescriptions
        p["last_edited"]   = datetime.datetime.now().isoformat()
        db[subject_id]     = p
        save_db(db)
        print("  ✓ Added and saved successfully!")

    elif choice == '3' and prescriptions:
        try:
            idx = int(input(f"  Enter medicine index to remove (1-{len(prescriptions)}): ").strip()) - 1
            if 0 <= idx < len(prescriptions):
                del prescriptions[idx]
                p["prescriptions"] = prescriptions
                db[subject_id]     = p
                save_db(db)
                print("  ✓ Medicine deleted!")
        except Exception:
            pass

    input("\n  Press ENTER to return to camera...")
    return db


# ── Open Camera Source Prompt ─────────────────────────────────────────────────
def prompt_cam_source():
    print("\n  Enter camera index (0, 1) or IP stream URL (http://...:8080/video)")
    print("  Or press Enter for default webcam [0]:")
    idx_str = input("  > ").strip()
    if idx_str.startswith("http://") or idx_str.startswith("https://") or idx_str.startswith("rtsp://"):
        return idx_str
    return int(idx_str) if idx_str.isdigit() else 0


# ── Main Dispenser Application ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SMART MEDICINE DISPENSER — Patient Face Recognition")
    print("=" * 60)

    db            = load_db()
    embeddings, _ = load_or_train(db)
    detector      = make_detector((640, 480))
    recognizer    = make_recognizer()

    cam_source = prompt_cam_source()
    stream     = VideoStream(cam_source)

    if not stream.isOpened():
        print(f"[ERROR] Could not open camera source: {cam_source}")
        sys.exit(1)

    # Initialize OpenCV Window for 480p
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    print("\n[READY] Kiosk HUD active.")
    print("  Keys: C=Capture  E=Edit  K=Lock/Unlock  SPC=Freeze  F=Fullscreen  Q=Quit\n")

    frozen                = False
    frozen_frame          = None
    fullscreen            = False
    cur_patient           = None
    cur_status            = "UNKNOWN"
    cur_dist              = 0.0
    locked                = False
    locked_patient        = None
    last_eval_time        = 0.0
    latest_rect           = None

    # Inference loop
    while True:
        # 1. Grab latest frame from zero-latency stream
        if not frozen:
            ret, raw = stream.read()
            if not ret or raw is None:
                time.sleep(0.02)
                continue
            frame = cv2.flip(raw, 1)
        else:
            frame = frozen_frame.copy()

        now = time.time()
        eval_countdown = max(0.0, 1.0 - (now - last_eval_time))

        # 2. Strict 1-Second Inference Tick (Prevents CPU lag & stream buffer buildup)
        if not frozen and (now - last_eval_time >= 1.0):
            last_eval_time = now
            # Detect face on current sample
            rect = detect_largest_face(frame, detector)
            latest_rect = rect

            if rect is not None:
                aligned = recognizer.alignCrop(frame, rect)
                feat = recognizer.feature(aligned)

                best_id = None
                best_score = 0.0
                dis_type = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
                for pid, db_feat in embeddings.items():
                    score = recognizer.match(feat, db_feat, dis_type)
                    if score > best_score:
                        best_score = score
                        best_id = pid

                cur_dist = best_score
                if best_id and best_score >= MATCH_THRESHOLD:
                    cur_patient = best_id
                    cur_status  = "AUTHORIZED" if best_score >= CAUTION_THRESHOLD else "LOW_CONFIDENCE"
                else:
                    cur_patient = None
                    cur_status  = "UNKNOWN"
            else:
                cur_patient = None
                cur_status  = "UNKNOWN"
                cur_dist    = 0.0

            if hw:
                hw.update_status(cur_status, locked)

        # 3. Render 480p Dashboard
        display_patient_id = locked_patient if locked else cur_patient
        display_status     = "LOCKED"       if locked else cur_status
        patient_data       = db.get(display_patient_id) if display_patient_id else None

        dashboard = render_480p_dashboard(
            frame=frame,
            rect=latest_rect,
            patient=patient_data,
            dist=cur_dist,
            status=display_status,
            locked=locked,
            is_frozen=frozen,
            eval_countdown=eval_countdown
        )

        cv2.imshow(WINDOW_NAME, dashboard)

        # 4. Handle Key Inputs
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord(' '):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()
                print("[FREEZE] Frame paused.")
            else:
                print("[FREEZE] Live feed resumed.")

        elif key == ord('f') or key == ord('F'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
            if not fullscreen:
                cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

        elif key == ord('k') or key == ord('K'):
            if locked:
                print(f"[LOCK] Unlocked patient {locked_patient}.")
                locked         = False
                locked_patient = None
            elif cur_patient:
                locked         = True
                locked_patient = cur_patient
                print(f"[LOCK] Locked onto patient {locked_patient}. Press K to unlock.")
            else:
                print("[LOCK] No patient recognized to lock.")

            if hw:
                hw.update_status(cur_status, locked)

        elif key == ord('e') or key == ord('E'):
            target = locked_patient if locked else cur_patient
            if target:
                cv2.destroyAllWindows()
                db = edit_prescription_cli(target, db)
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
                last_eval_time = 0.0 # re-evaluate immediately
            else:
                print("[EDIT] Please recognize or lock a patient first.")

        elif key == ord('c') or key == ord('C'):
            pid = locked_patient if locked else cur_patient
            if not pid:
                cv2.destroyAllWindows()
                pid = input("\n  Enter Patient ID to enroll / capture: ").strip()
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

            if pid and pid in db:
                saved = capture_webcam_frames(stream, detector, pid, n_frames=10)
                if saved > 0:
                    print(f"[TRAIN] Updating embeddings for {pid} ...")
                    embeddings, _ = train(db)
                    print("[TRAIN] Recognition updated.")
            else:
                print(f"  [!] Patient ID '{pid}' not found.")
            frozen = False

        elif key == ord('r') or key == ord('R'):
            print("\n[RETRAIN] Re-training all patient embeddings ...")
            db = load_db()
            embeddings, _ = train(db)
            cur_patient = None
            cur_status  = "UNKNOWN"
            cur_dist    = 0.0
            print("[RETRAIN] Done.")

    stream.release()
    cv2.destroyAllWindows()
    if hw:
        hw.cleanup()
    print("[DONE] Dispenser session ended.")


if __name__ == "__main__":
    main()
