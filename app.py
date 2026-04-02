import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.capture_faces import append_capture_log, sanitize_mobile, sanitize_name, upsert_employee
from src.mark_attendance import (
    ATTENDANCE_DIR,
    CONFIDENCE_THRESHOLD,
    MODEL_FILE,
    LABELS_FILE,
    ensure_today_file,
    mark_attendance,
)
from src.train_model import MODELS_DIR, DATA_DIR, preprocess_face


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def get_training_assets():
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        raise FileNotFoundError("Model or labels missing. Train the model first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    with LABELS_FILE.open("r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    return recognizer, labels


def decode_image(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def extract_face(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = gray[y : y + h, x : x + w]
    return preprocess_face(face)


def save_face_sample(person_folder: Path, face_image: np.ndarray) -> Path:
    person_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"{person_folder.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    file_path = person_folder / file_name
    cv2.imwrite(str(file_path), face_image)
    return file_path


def load_employee_count() -> int:
    employees_file = DATA_DIR / "employees.csv"
    if not employees_file.exists():
        return 0
    with employees_file.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_recent_attendance(limit: int = 10):
    if not ATTENDANCE_DIR.exists():
        return []

    files = sorted(ATTENDANCE_DIR.glob("attendance_*.csv"), reverse=True)
    if not files:
        return []

    latest = files[0]
    with latest.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit][::-1]


def registration_section():
    st.subheader("Register employee")
    st.write("Capture a browser photo, save face samples, and register the employee record.")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Employee name", placeholder="om")
        mobile = st.text_input("Mobile number", placeholder="9876543210")
        employee_id = st.text_input("Employee ID", placeholder="EMP001")
    with col2:
        role = st.text_input("Role", value="Employee")
        company_name = st.text_input("Company name", value="Vickhardth Automation")

    photo = st.camera_input("Capture face photo")
    submitted = st.button("Register and save sample")

    if not submitted:
        return

    if not name.strip() or not mobile.strip() or not employee_id.strip():
        st.error("Name, mobile, and employee ID are required.")
        return

    image = decode_image(photo)
    if image is None:
        st.error("Please capture a face photo before submitting.")
        return

    face = extract_face(image)
    if face is None:
        st.error("No face detected in the captured image.")
        return

    clean_name = sanitize_name(name)
    clean_mobile = sanitize_mobile(mobile)
    folder_name = f"{clean_name}_{clean_mobile}"
    person_dir = DATA_DIR / folder_name

    try:
        upsert_employee(name.strip(), clean_mobile, employee_id.strip(), role.strip(), company_name.strip(), "")
    except ValueError as exc:
        st.error(str(exc))
        return

    saved_path = save_face_sample(person_dir, face)
    append_capture_log(name.strip(), clean_mobile, folder_name, 1)

    st.success(f"Saved face sample to {saved_path}")
    st.info("Capture multiple samples by repeating this step before training.")


def training_section():
    st.subheader("Train model")
    st.write("This builds the face recognizer from images stored in `data/`.")

    if st.button("Train model"):
        try:
            from src.train_model import load_training_data

            images, labels, label_map = load_training_data()
            if not images:
                st.error("No training images found in `data/`.")
                return

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, labels)
            recognizer.save(str(MODEL_FILE))

            with LABELS_FILE.open("w", encoding="utf-8") as f:
                json.dump(label_map, f, indent=2)

            st.success(f"Model saved to {MODEL_FILE}")
            st.success(f"Labels saved to {LABELS_FILE}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")


def attendance_section():
    st.subheader("Mark attendance from photo")
    st.write("Upload a face photo or use the browser camera to mark IN/OUT.")

    uploaded = st.camera_input("Capture attendance photo")
    if st.button("Recognize and mark attendance"):
        try:
            recognizer, labels = get_training_assets()
        except Exception as exc:
            st.error(str(exc))
            return

        image = decode_image(uploaded)
        if image is None:
            st.error("Please capture a photo first.")
            return

        face = extract_face(image)
        if face is None:
            st.error("No face detected in the captured photo.")
            return

        label_id, confidence = recognizer.predict(face)
        if confidence >= CONFIDENCE_THRESHOLD:
            st.error(f"Face not recognized. Confidence: {confidence:.1f}")
            return

        name = labels.get(label_id, "unknown")
        today_file = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
        ensure_today_file(today_file)
        message, ok = mark_attendance(name, today_file)
        if ok:
            st.success(message)
        else:
            st.warning(message)


def attendance_table_section():
    st.subheader("Latest attendance")
    rows = load_recent_attendance()
    if not rows:
        st.caption("No attendance file found yet.")
        return
    st.dataframe(rows, use_container_width=True)


def main():
    st.set_page_config(page_title="Vickhardth Attendance", page_icon="VA", layout="wide")

    st.title("Vickhardth Attendance")
    st.caption("Streamlit control panel for registration, training, and attendance.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Employees", load_employee_count())
    col2.metric("Training data", len(list(DATA_DIR.glob("*/*.jpg"))) if DATA_DIR.exists() else 0)
    col3.metric("Model ready", "Yes" if MODEL_FILE.exists() and LABELS_FILE.exists() else "No")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Register", "Train", "Attendance", "Records"]
    )
    with tab1:
        registration_section()
    with tab2:
        training_section()
    with tab3:
        attendance_section()
    with tab4:
        attendance_table_section()


if __name__ == "__main__":
    main()
