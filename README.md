# AI Attendance System

An AI-powered desktop attendance system that uses face detection and face recognition to automate student attendance.

## Features

- Student enrollment with webcam image capture
- Automatic face encoding generation
- Real-time face detection and recognition
- Attendance marking by subject
- Excel export for attendance records
- Simple desktop interface built with Tkinter
- Basic evaluation script for recognition performance

## Tech Stack

- Python
- OpenCV
- face_recognition
- dlib
- Ultralytics YOLOv8 Face
- NumPy
- Pandas
- OpenPyXL
- Scikit-learn
- Matplotlib
- Seaborn
- Tkinter

## Project Structure

```text
Attendance System/
├── Code/
│   ├── UI.py
│   ├── index.py
│   ├── enroll.py
│   ├── recognize.py
│   ├── encode_faces.py
│   ├── attendance.py
│   ├── evaluation.py
│   ├── config.py
│   └── requirements.txt
└── Project_SS/
    ├── Attendance/
    └── Enrollment/
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r Code/requirements.txt
```

3. Download the YOLO model files if they are not already present:

- `yolov8n.pt`
- `yolov8n-face.pt`

4. Run the desktop UI:

```bash
python Code/UI.py
```

## How It Works

### Enrollment

- Capture student face images from webcam
- Save images inside a student folder
- Store SAP ID in `info.txt`
- Generate face encodings for recognition

### Attendance

- Detect faces in webcam frames using YOLOv8 face detection
- Recognize faces against enrolled encodings
- Mark students present once per session
- Export attendance to Excel by subject

## Important Privacy Note

This project uses face images and derived face encodings. Do not upload private student datasets or generated biometric encodings to a public repository.

Recommended exclusions:

- `Code/dataset/`
- `Code/encodings.pickle`
- `Code/attendance_excel/`

## Future Improvements

- Multi-frame confirmation before marking attendance
- Liveness detection to reduce spoofing
- Confidence calibration and false-positive reduction
- Database integration
- Web dashboard and analytics
- Role-based access control
- Cloud deployment

## Screenshots

You can add screenshots from the `Project_SS/` folder here after publishing the repository.

## Author

Add your name, LinkedIn, and GitHub profile here.
