# 🎯 AI-Based Attendance System

An **AI-powered Attendance Management System** that uses **Face Recognition** to automatically mark attendance.  
Built using **Flask**, **OpenCV**, and **Face Recognition (dlib)** libraries.

---

## 🧠 Overview

This project captures real-time images through a webcam, detects and recognizes faces, and records attendance data (Name, Date, Time) automatically into a CSV file.

### ✨ Key Features
- 👤 Face Detection & Recognition using OpenCV and dlib  
- 🧾 Automatic attendance marking (stored in CSV format)  
- 🌐 Web interface built with Flask  
- 📸 Capture and store new faces  
- 📅 View attendance reports easily  

---

## 📁 Project Structure

AI_BASED_ATTENDANCE_SYSTEM-main/
│
├── app.py # Main Flask app
├── haarcascade_frontalface_default.xml # Face detection model
├── Attendance/ # Stores attendance CSV files
├── face_recognition_flask/ # Core face recognition module
├── static/ # Static assets (CSS, JS, images)
├── background.png # UI background image
├── Capture.PNG # Capture button image
└── README.md # Project documentation

---

## ⚙️ Installation

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/yourusername/AI_BASED_ATTENDANCE_SYSTEM.git
cd AI_BASED_ATTENDANCE_SYSTEM-main


2.Install Python Dependencies

Make sure you have Python 3.8+ installed.

Install required libraries:
pip install flask opencv-python face-recognition numpy pandas
If face-recognition fails to install due to dlib, follow these steps:

Install CMake
 (add to PATH)

Install Visual Studio Build Tools

Then re-run:
pip install dlib face-recognition
OR use this quick fix:
pip install cmake dlib-bin face-recognition

Running the Project

Run the Flask app:
python app.py

You should see:
 * Running on http://127.0.0.1:5000/

Open your browser and go to:
👉 http://127.0.0.1:5000/

🧩 How It Works

Register Faces
Capture images of users’ faces and store encodings.

Start Attendance
The camera opens and automatically detects and recognizes faces.

Mark Attendance
When a known face is detected, name, date, and time are recorded in:
/Attendance/Attendance_<date>.csv
View Attendance
Open the CSV file to view attendance records.

Dependencies
Package	Purpose
Flask	Web Framework
OpenCV	Image Capture & Face Detection
dlib	Facial Landmark & Recognition
face-recognition	Easy face encoding & comparison
numpy	Array operations
pandas	CSV handling
🚀 Future Enhancements

Cloud-based data storage (Firebase / AWS)

Admin dashboard for analytics

SMS/Email notifications

Multi-camera support

👨‍💻 Author

Your Name
📧 your.email@example.com

🌐 GitHub Profile

📜 License

This project is open-source and available under the MIT License
.
---

Would you like me to **personalize** this README (add your name, GitHub link, and local path instructions for Windows)?  
I can also generate a version in **docx or PDF format** if you need it for a project submission.







