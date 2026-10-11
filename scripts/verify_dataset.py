"""
verify_dataset.py
-----------------
Verifies the integrity of the entire generated patient dataset.
Checks the JSON database, the mapping CSV, and every physical image file.
Works for any number of patients (not hardcoded).
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
DB_PATH = METADATA_DIR / "patient_database.json"
MAP_CSV_PATH = METADATA_DIR / "patient_face_map.csv"

REQUIRED_FIELDS = ["subject_id", "gender", "anchor_age", "prescription", "enrolled", "face_image_path"]
REQUIRED_PRESCRIPTION_FIELDS = ["drug", "dose_val", "dose_unit", "route", "doses_per_24_hrs"]

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

def check(condition, pass_msg, fail_msg):
    if condition:
        print(f"  {PASS} {pass_msg}")
    else:
        print(f"  {FAIL} {fail_msg}")
        errors.append(fail_msg)
    return condition

def verify():
    print("=" * 60)
    print("  Dataset Verification Report")
    print("=" * 60)

    # ── 1. Check patient_database.json ──────────────────────────
    print("\n[1] Checking patient_database.json ...")
    if not check(DB_PATH.exists(), "patient_database.json exists", "patient_database.json NOT FOUND"):
        print("\nCannot continue — critical file missing.")
        return

    with open(DB_PATH) as f:
        db = json.load(f)

    total_patients = len(db)
    check(total_patients > 0, f"Database has {total_patients} entries", "Database is empty")

    # ── 2. Check required fields in every record ─────────────────
    print(f"\n[2] Validating all {total_patients} patient records ...")
    missing_fields_count = 0
    missing_rx_fields_count = 0
    for pid, data in db.items():
        for field in REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"Patient {pid} missing field '{field}'")
                missing_fields_count += 1
        if "prescription" in data:
            for rxf in REQUIRED_PRESCRIPTION_FIELDS:
                if rxf not in data["prescription"]:
                    errors.append(f"Patient {pid} prescription missing '{rxf}'")
                    missing_rx_fields_count += 1

    check(missing_fields_count == 0,
          "All patient records have required fields",
          f"{missing_fields_count} missing field(s) across records")
    check(missing_rx_fields_count == 0,
          "All prescription records have required fields",
          f"{missing_rx_fields_count} missing prescription field(s)")

    # ── 3. Check patient_face_map.csv ───────────────────────────
    print(f"\n[3] Checking patient_face_map.csv ...")
    if not check(MAP_CSV_PATH.exists(), "patient_face_map.csv exists", "patient_face_map.csv NOT FOUND"):
        pass
    else:
        df = pd.read_csv(MAP_CSV_PATH)
        check(len(df) == total_patients,
              f"Map CSV has {len(df)} rows matching {total_patients} patients",
              f"Map CSV has {len(df)} rows but expected {total_patients}")
        check(len(df["subject_id"].unique()) == len(df),
              "No duplicate subject_ids in map CSV",
              "Duplicate subject_ids found in map CSV")

    # ── 4. Check physical image files ───────────────────────────
    print(f"\n[4] Checking physical image files on disk ...")
    missing_files = []
    corrupt_files = []
    tiny_files = []

    for pid, data in db.items():
        img_path = PROJECT_ROOT / data["face_image_path"]
        if not img_path.exists():
            missing_files.append(pid)
            continue
        # Check file is readable by OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            corrupt_files.append(pid)
            continue
        # Check image is not suspiciously tiny (< 20x20 pixels)
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            tiny_files.append(pid)

    check(len(missing_files) == 0,
          f"All {total_patients} image files exist on disk",
          f"{len(missing_files)} image files MISSING: {missing_files[:5]}")
    check(len(corrupt_files) == 0,
          "All image files are readable (not corrupt)",
          f"{len(corrupt_files)} corrupt/unreadable image(s): {corrupt_files[:5]}")
    check(len(tiny_files) == 0,
          "All images have reasonable dimensions (>=20x20px)",
          f"{len(tiny_files)} suspiciously small image(s): {tiny_files[:5]}")

    # ── 5. Check subject_id consistency ─────────────────────────
    print(f"\n[5] Checking cross-consistency between DB and CSV ...")
    db_ids = set(db.keys())
    if MAP_CSV_PATH.exists():
        df = pd.read_csv(MAP_CSV_PATH)
        csv_ids = set(df["subject_id"].astype(str))
        only_in_db = db_ids - csv_ids
        only_in_csv = csv_ids - db_ids
        check(len(only_in_db) == 0,
              "All DB patients appear in the map CSV",
              f"{len(only_in_db)} patient(s) in DB but missing from CSV: {list(only_in_db)[:5]}")
        check(len(only_in_csv) == 0,
              "All CSV rows have a matching DB record",
              f"{len(only_in_csv)} CSV row(s) with no matching DB record: {list(only_in_csv)[:5]}")

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not errors:
        print(f"  RESULT: ALL CHECKS PASSED — {total_patients} patients fully verified.")
    else:
        print(f"  RESULT: {len(errors)} ERROR(S) FOUND:")
        for e in errors:
            print(f"    - {e}")
    print("=" * 60)

if __name__ == "__main__":
    verify()
