# Face Recognition Attendance System (Python + OpenCV)

This project records attendance using live face recognition from webcam input.
The desktop Tkinter GUI is now the primary app.

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Capture face samples

Capture images for each person:

```powershell
python src/capture_faces.py --name om --samples 40
python src/capture_faces.py --name rahul --samples 40
```

Captured images are saved in `data/<person_name>/`.

## 3) Train recognizer

```powershell
python src/train_model.py
```

This creates:
- `models/face_trainer.yml`
- `models/labels.json`

## 4) Start attendance

```powershell
python src/mark_attendance.py
```

Attendance is saved in `attendance/attendance_YYYYMMDD.csv`.

## Desktop GUI app

Use the Tkinter desktop app for registration and attendance:

```powershell
python src/gui_app.py
# or
python main.py
```

In the GUI:
- Fill employee name and mobile.
- Click `Register & Auto Train`.
- Attendance starts automatically when a trained model exists.

## Notes

- Press `q` in the camera window to stop.
- If recognition is weak, increase image samples and retrain.
- You can tune `CONFIDENCE_THRESHOLD` in `src/mark_attendance.py` (lower is stricter).
- Location is fetched from the device/network on the desktop app, so it is more accurate than the old Streamlit cloud location.
