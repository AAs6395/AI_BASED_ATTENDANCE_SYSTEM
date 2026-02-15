import cv2
import os
import logging
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# -------------------- LOGGING (SAFE REPLACEMENT FOR PRINT) --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
FACES_DIR = os.path.join(STATIC_DIR, "faces")

# -------------------- FLASK APP --------------------
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# -------------------- DIRECTORIES --------------------
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# -------------------- VARIABLES --------------------
nimgs = 10

imgBackground = cv2.imread(os.path.join(BASE_DIR, "background.png"))

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

attendance_file = os.path.join(
    ATTENDANCE_DIR,
    f"Attendance-{datetoday}.csv"
)

face_detector = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)

# -------------------- ENSURE CSV EXISTS --------------------
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Roll", "Time"]).to_csv(
        attendance_file, index=False
    )

# -------------------- FUNCTIONS --------------------
def totalreg():
    return len(os.listdir(FACES_DIR))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, 1.2, 5, minSize=(20, 20)
        )
        return faces
    except:
        return []

def identify_face(facearray):
    model = joblib.load(
        os.path.join(STATIC_DIR, "face_recognition_model.pkl")
    )
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []

    for user in os.listdir(FACES_DIR):
        user_path = os.path.join(FACES_DIR, user)
        for imgname in os.listdir(user_path):
            img = cv2.imread(os.path.join(user_path, imgname))
            img = cv2.resize(img, (50, 50))
            faces.append(img.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)

    joblib.dump(
        knn,
        os.path.join(STATIC_DIR, "face_recognition_model.pkl")
    )

def extract_attendance():
    if not os.path.exists(attendance_file):
        return [], [], [], 0

    df = pd.read_csv(attendance_file)
    return df["Name"], df["Roll"], df["Time"], len(df)

def add_attendance(name):
    username, userid = name.split("_")
    time_now = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)
    if int(userid) not in list(df["Roll"]):
        with open(attendance_file, "a") as f:
            f.write(f"\n{username},{userid},{time_now}")

def getallusers():
    userlist = os.listdir(FACES_DIR)
    names = []
    rolls = []

    for user in userlist:
        name, roll = user.split("_")
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, len(userlist)

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    names, rolls, times, l = extract_attendance()

    # ✅ SAFE DEBUG (NO Errno 22)
    logging.info(f"Template directory in use: {TEMPLATE_DIR}")

    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2
    )

@app.route("/start", methods=["GET"])
def start():
    if "face_recognition_model.pkl" not in os.listdir(STATIC_DIR):
        return home()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                identified_person,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        imgBackground[162:162+480, 55:55+640] = frame
        cv2.imshow("Attendance", imgBackground)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return home()

@app.route("/add", methods=["POST"])
def add():
    newusername = request.form["newusername"]
    newuserid = request.form["newuserid"]

    userfolder = os.path.join(FACES_DIR, f"{newusername}_{newuserid}")
    os.makedirs(userfolder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0

    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            if j % 5 == 0 and i < nimgs:
                cv2.imwrite(
                    os.path.join(userfolder, f"{i}.jpg"),
                    frame[y:y+h, x:x+w]
                )
                i += 1
            j += 1

        cv2.imshow("Register User", frame)
        if i >= nimgs or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()
    return home()

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
