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