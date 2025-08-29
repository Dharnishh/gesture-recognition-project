<<<<<<< HEAD
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import random
import os  

app = Flask(__name__) 

# Load the gesture recognition model
model = load_model('gesture_model.h5')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# List of commands for gestures
commands = ['sit', 'stand', 'fly']
attempts = 10
current_attempt = 0
score = 0
current_command = None  # Track the current command

def generate_gesture_frames():
    global score, current_command
    cap = cv2.VideoCapture(0)  # Ensure the correct camera index is used

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = [landmark for lm in results.pose_landmarks.landmark for landmark in [lm.x, lm.y, lm.z]]
            if len(landmarks) > 0:
                prediction = model.predict(np.array([landmarks]).reshape(1, 1, -1))
                gesture_index = np.argmax(prediction)
                recognized_gesture = commands[gesture_index]
                cv2.putText(frame, f'Gesture: {recognized_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check if the recognized gesture matches the current command
                if recognized_gesture == current_command:
                    print(f"Correct Gesture: {recognized_gesture}! Incrementing score...")
                    score += 1  # Increment score for correct gesture
                    current_command = None  # Reset the command after the correct gesture
                else:
                    print(f"Incorrect Gesture: {recognized_gesture}. Expected: {current_command}")

            else:
                cv2.putText(frame, 'No landmarks detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/get_command')
def get_command():
    global current_attempt, score, game_active, current_command
    if current_attempt < attempts:
        current_command = random.choice(commands)  # Set a new random command
        current_attempt += 1
        return jsonify({'command': current_command, 'attempts_left': attempts - current_attempt})
    else:
        game_active = False  # End the game when attempts are finished
        return jsonify({'command': 'Game Over', 'score': score, 'attempts_left': 0})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    global current_attempt, score, current_command
    current_attempt = 0
    score = 0
    current_command = None
    return redirect(url_for('game'))

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/video_feed')
def video_feed():
    print("Video feed route accessed.")  # Log to check if the route is called
    return Response(generate_gesture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit_score', methods=['POST'])
def submit_score():
    global score
    score += 1  # Increment score for correct gesture recognition
    return jsonify({'score': score})

@app.route('/score-card')
def score_card():
    print(f'Current Score: {score}')  # Debugging output
    return render_template('score_card.html', score=score,total=attempts)

@app.route('/upload-model')
def upload_model():
    return render_template('upload_model.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'model_file' not in request.files:
        return redirect(url_for('upload_model'))

    file = request.files['model_file']
    if file.filename == '':
        return redirect(url_for('upload_model'))

    if file and file.filename.endswith('.h5'):
        file_path = os.path.join('models', file.filename)
        file.save(file_path)
        # Optionally load the new model if needed
        # model = load_model(file_path)
        return redirect(url_for('index'))  # Redirect to index after upload

    return redirect(url_for('upload_model'))

if __name__ == '__main__':
    app.run(debug=True)
=======
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf

# ----------------------------
# Load YOLO model
# ----------------------------
yolo_model = YOLO("yolov8n.pt")  # person detection

# ----------------------------
# Load Gesture Recognition Model (hardcoded path)
# ----------------------------
gesture_model = tf.keras.models.load_model("gesture_model.h5")

# Gesture Labels (update these with your model's classes)
gesture_labels = ["Hello", "Yes", "No", "Thanks", "Stop"]

# ----------------------------
# Scoreboard dictionary
# ----------------------------
scores = {}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎥 Real-time Multi-person Gesture Recognition")
st.markdown("YOLO detects persons, Gesture model classifies their actions")

start_btn = st.button("Start Detection")

# Video frame output
frame_window = st.image([])
score_placeholder = st.empty()

# ----------------------------
# Helper: Update Scoreboard
# ----------------------------
def update_scoreboard(scores_dict):
    score_text = "### 📊 Live Scoreboard\n"
    if not scores_dict:
        score_text += "_No persons detected yet._"
    else:
        for pid, actions in scores_dict.items():
            score_text += f"- **Person {pid}** → {actions}\n"
    return score_text

# ----------------------------
# Main Loop
# ----------------------------
if start_btn:
    cap = cv2.VideoCapture(0)  # use 0 for webcam
    
    if not cap.isOpened():
        st.error("⚠️ Failed to read from camera. Try restarting Streamlit or check camera permissions.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            # YOLO person detection
            results = yolo_model(frame, classes=[0], conf=0.5, verbose=False)  # class 0 = person

            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

            # Loop over detected persons
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                person_id = idx + 1

                # Crop person ROI
                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size > 0:
                    # Preprocess for gesture model
                    resized = cv2.resize(person_roi, (64, 64))  # match your model input size
                    resized = resized.astype("float32") / 255.0
                    resized = np.expand_dims(resized, axis=0)

                    # Predict gesture
                    try:
                        pred = gesture_model.predict(resized, verbose=0)
                        gesture_id = np.argmax(pred)
                        gesture_name = gesture_labels[gesture_id]
                    except Exception as e:
                        gesture_name = "Unknown"

                    # Update scoreboard
                    if person_id not in scores:
                        scores[person_id] = []
                    scores[person_id].append(gesture_name)

                    # Draw bounding box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{person_id} {gesture_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)

            # Update Scoreboard
            score_placeholder.markdown(update_scoreboard(scores))

    cap.release()
>>>>>>> ba1f0d01e794f929331083a464713dd5805c93a7
