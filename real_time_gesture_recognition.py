import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the model
model = load_model('gesture_model.h5')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to extract landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = [landmark for lm in results.pose_landmarks.landmark for landmark in [lm.x, lm.y, lm.z]]
        return landmarks
    return None

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame)
    if landmarks:
        # Reshape landmarks to match model input
        prediction = model.predict(np.array([landmarks]).reshape(1, 1, -1))
        gesture_index = np.argmax(prediction)
        # Display the predicted gesture on the frame
        cv2.putText(frame, f'Gesture: {gesture_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()