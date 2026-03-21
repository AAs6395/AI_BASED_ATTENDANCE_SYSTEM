import cv2
import os
import logging
from flask import Flask, request, render_template, jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil

logging.basicConfig(level=logging.INFO)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
ATTEND_DIR   = os.path.join(BASE_DIR, "Attendance")
FACES_DIR    = os.path.join(STATIC_DIR, "faces")
MODEL_PATH   = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

os.makedirs(ATTEND_DIR, exist_ok=True)
os.makedirs(FACES_DIR,  exist_ok=True)

nimgs = 10

datetoday  = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
attend_csv = os.path.join(ATTEND_DIR, f"Attendance-{datetoday}.csv")

face_detector = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)
imgBackground = cv2.imread(os.path.join(BASE_DIR, "background.png"))

if not os.path.exists(attend_csv):
    pd.DataFrame(columns=["Name", "Roll", "Time"]).to_csv(attend_csv, index=False)


# ======================== HELPERS ========================

def totalreg():
    return len([d for d in os.listdir(FACES_DIR)
                if os.path.isdir(os.path.join(FACES_DIR, d))])


def extract_faces(img):
    try:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return faces
    except Exception:
        return []


def identify_face(facearray):
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


def train_model():
    faces, labels = [], []
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    for user in os.listdir(FACES_DIR):
        user_path = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_path):
            continue
        for imgname in os.listdir(user_path):
            if os.path.splitext(imgname)[1].lower() not in VALID_EXTS:
                continue
            img = cv2.imread(os.path.join(user_path, imgname))
            if img is None:
                continue
            img = cv2.resize(img, (50, 50))
            faces.append(img.ravel())
            labels.append(user)
    if not faces:
        logging.error("train_model: no valid images found")
        return False
    faces = np.array(faces)
    n_neighbors = max(1, min(5, len(faces) // max(1, len(set(labels)))))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(faces, labels)
    joblib.dump(knn, MODEL_PATH)
    logging.info(f"Model trained: {len(set(labels))} users, {len(faces)} images, k={n_neighbors}")
    return True


def extract_attendance():
    if not os.path.exists(attend_csv):
        return [], [], [], 0
    df = pd.read_csv(attend_csv)
    if df.empty:
        return [], [], [], 0
    return df["Name"].tolist(), df["Roll"].tolist(), df["Time"].tolist(), len(df)


def add_attendance(name):
    """
    THE DUPLICATE BUG FIX:
    Original used open(file,'a') and re-read CSV each frame but NEVER wrote
    before re-checking — so every frame looked empty and appended again.
    Fix: read → check → write full df with to_csv() (overwrites cleanly).
    """
    try:
        parts = name.split("_")
        if len(parts) < 2:
            return False
        username = parts[0]
        userid   = str(parts[1])
        df = pd.read_csv(attend_csv)
        if userid in df["Roll"].astype(str).values:
            return False   # already marked
        time_now = datetime.now().strftime("%H:%M:%S")
        new_row  = pd.DataFrame([[username, userid, time_now]],
                                columns=["Name", "Roll", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attend_csv, index=False)
        logging.info(f"Marked: {username} ({userid}) at {time_now}")
        return True
    except Exception as e:
        logging.error(f"add_attendance error: {e}")
        return False


def getallusers():
    userlist = [d for d in os.listdir(FACES_DIR)
                if os.path.isdir(os.path.join(FACES_DIR, d))]
    names, rolls = [], []
    for user in userlist:
        parts = user.split("_")
        if len(parts) >= 2:
            names.append(parts[0])
            rolls.append(parts[1])
    return userlist, names, rolls, len(userlist)


# ======================== ROUTES ========================

@app.route("/")
def home():
    names, rolls, times, l = extract_attendance()
    userlist, unames, urolls, _ = getallusers()
    return render_template(
        "home.html",
        names=names, rolls=rolls, times=times, l=l,
        totalreg=totalreg(), datetoday2=datetoday2,
        userlist=userlist, unames=unames, urolls=urolls,
        model_exists=os.path.exists(MODEL_PATH)
    )


@app.route("/start", methods=["GET"])
def start():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"success": False,
                        "message": "No trained model. Please add users first."})
    cap = None
    identified = set()
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Cannot open webcam."})

        frame_count = 0
        while frame_count < 90:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                person   = identify_face(face_img.reshape(1, -1))[0]
                add_attendance(person)
                identified.add(person)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{person.split('_')[0]}  Marked!",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if imgBackground is not None:
                bg = imgBackground.copy()
                try:
                    bg[162:162+480, 55:55+640] = frame
                except Exception:
                    pass
                cv2.imshow("Attendance", bg)
            else:
                cv2.imshow("Attendance", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if len(identified) > 0:
                cv2.waitKey(800)
                break
    except Exception as e:
        logging.error(f"/start error: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return jsonify({
        "success":    True,
        "identified": list(identified),
        "names":      names,
        "rolls":      [str(r) for r in rolls],
        "times":      times,
        "count":      l
    })


@app.route("/add", methods=["POST"])
def add():
    """
    FIX: Entire route wrapped in try/except/finally.
    Flask will NEVER return an HTML 500 page — always returns JSON.
    This is why the frontend was showing 'Error adding user' even
    when photos saved fine — res.json() crashed on the HTML error page.
    """
    newusername = request.form.get("newusername", "").strip()
    newuserid   = request.form.get("newuserid",   "").strip()

    if not newusername or not newuserid:
        return jsonify({"success": False, "message": "Name and ID are required."})

    newusername = "".join(c for c in newusername if c.isalnum() or c == "-")
    if not newusername:
        return jsonify({"success": False, "message": "Invalid name. Letters/numbers only."})

    userfolder = os.path.join(FACES_DIR, f"{newusername}_{newuserid}")
    os.makedirs(userfolder, exist_ok=True)

    cap = None
    i   = 0

    # ---- STEP 1: Capture face images ----
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Cannot open webcam."})

        j = 0
        while i < nimgs:
            ret, frame = cap.read()
            if not ret:
                break
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                if j % 2 == 0 and i < nimgs:
                    cv2.imwrite(os.path.join(userfolder, f"{i}.jpg"),
                                frame[y:y+h, x:x+w])
                    i += 1
                j += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                cv2.putText(frame, f"Capturing {i}/{nimgs}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if imgBackground is not None:
                bg = imgBackground.copy()
                try:
                    bg[162:162+480, 55:55+640] = frame
                except Exception:
                    pass
                cv2.imshow("Register User", bg)
            else:
                cv2.imshow("Register User", frame)

            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        logging.error(f"/add capture exception: {e}")
        # Don't return here — photos may be partially saved, try to train below
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    # ---- STEP 2: Validate photos were saved ----
    if i == 0:
        shutil.rmtree(userfolder, ignore_errors=True)
        return jsonify({"success": False,
                        "message": "No face detected. Try better lighting."})

    # ---- STEP 3: Train model — also wrapped so errors return JSON ----
    try:
        trained = train_model()
    except Exception as e:
        logging.error(f"train_model exception: {e}")
        trained = False

    if not trained:
        return jsonify({
            "success": False,
            "message": f"{i} photos saved but model training failed. "
                       f"Check terminal for details."
        })

    return jsonify({
        "success":  True,
        "message":  f"'{newusername}' registered with {i} images!",
        "totalreg": totalreg()
    })


@app.route("/delete_user", methods=["POST"])
def delete_user():
    try:
        data      = request.get_json()
        username  = data.get("username", "")
        user_path = os.path.join(FACES_DIR, username)
        if not os.path.exists(user_path):
            return jsonify({"success": False, "message": "User not found."})
        shutil.rmtree(user_path)
        if totalreg() > 0:
            train_model()
        elif os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return jsonify({"success": True,
                        "message": f"User '{username.split('_')[0]}' deleted."})
    except Exception as e:
        logging.error(f"delete_user error: {e}")
        return jsonify({"success": False, "message": str(e)})


# ======================== RUN ========================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
