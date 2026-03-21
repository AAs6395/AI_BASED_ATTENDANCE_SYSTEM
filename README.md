# 🎯 FaceAttend — AI-Based Attendance System

A smart, real-time attendance system powered by face recognition.  
Built with **Flask**, **OpenCV**, and **scikit-learn (KNN)**.

---

## ✨ Features

- 🎭 Real-time face detection & recognition via webcam
- 🤖 KNN-based face classifier (auto-trains when users are added)
- 📋 Auto-marks attendance into daily CSV files
- 🚫 Duplicate prevention — each person marked once per day
- 👥 Add / delete users from the web UI
- 📱 Modern dark UI with smooth animations

---

## 📁 Project Structure

```
attendance_system/
├── app.py                          # Main Flask app (all routes & logic)
├── requirements.txt                # Python dependencies
├── haarcascade_frontalface_default.xml  # OpenCV face detector
├── background.png                  # UI background (optional)
├── templates/
│   └── home.html                   # Frontend UI
├── static/
│   ├── faces/                      # Captured face images per user
│   └── face_recognition_model.pkl  # Trained KNN model (auto-generated)
└── Attendance/
    └── Attendance-MM_DD_YY.csv     # Daily attendance logs
```

---

## ⚙️ Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Windows, if `opencv-python` fails, try `pip install opencv-python-headless`.

### 2. Run the app

```bash
python app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000
```

---

## 🧩 How It Works

### Register a User
1. Enter the user's name and ID in the "Add New User" tab
2. Click **"Capture Face & Register"**
3. The webcam opens — look directly at the camera
4. 15 face images are captured automatically
5. The KNN model is retrained with the new user

### Take Attendance
1. Click **"Take Attendance Now"**
2. The webcam scans for ~5 seconds
3. Recognized faces are matched and recorded in today's CSV
4. Attendance table updates in real-time on the page

### View Attendance
- Today's attendance is shown in the left panel
- CSV files are stored in `/Attendance/` folder

---

## 🐛 Fixes Applied (from original version)

| Issue | Fix |
|-------|-----|
| `add_attendance` crash on duplicate roll | Fixed with proper string comparison |
| Model not found crash on `/start` | Returns JSON error with message |
| `extract_faces` returning wrong type | Returns empty list `[]` on error |
| Webcam not released on error | Added proper `cap.release()` handling |
| Form using page redirect (slow) | Converted to async fetch + JSON API |
| No feedback during webcam capture | Added loading overlay with messages |
| No delete user functionality | Added `/delete_user` route + UI button |
| No input validation | Added name/ID validation + sanitization |
| Hard crash if no model PKL | Graceful JSON error response |
| `home.html` was in wrong directory | Moved to `/templates/` correctly |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| OpenCV | Face detection & webcam |
| scikit-learn | KNN classifier |
| NumPy | Array operations |
| pandas | CSV handling |
| joblib | Model save/load |

---

## 👨‍💻 Author

Aashish Joshi — [jaashish109@gmail.com](mailto:jaashish109@gmail.com)
