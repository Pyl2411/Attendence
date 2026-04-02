# Face Recognition Attendance System (Python + OpenCV)

This project records attendance using live face recognition from webcam input.

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

## Optional: GUI app (easy for employees)

Use a simple desktop app with buttons for capture, train, and attendance:

```powershell
python src/gui_app.py
```

In the GUI:
- Fill employee name and mobile.
- Click `Register & Capture Face`.
- Click `Train Model`.
- Click `Start Attendance`.

## Streamlit deploy

For Streamlit Cloud, use this repo-relative main file path:

```text
app.py
```

Run locally with:

```powershell
streamlit run app.py
```

## Notes

- Press `q` in the camera window to stop.
- If recognition is weak, increase image samples and retrain.
- You can tune `CONFIDENCE_THRESHOLD` in `src/mark_attendance.py` (lower is stricter).
