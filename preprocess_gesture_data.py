import os
import cv2
import numpy as np
import mediapipe as mp

# Define gestures and setup directories
gestures = ['sit', 'stand', 'fly']
dataset_directory = 'gesture_data'
landmarks_data = []
labels = []

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to extract pose landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = [landmark for lm in results.pose_landmarks.landmark for landmark in [lm.x, lm.y, lm.z]]
        return landmarks
    return None

# Loop through all gesture data and extract landmarks
for gesture_index, gesture in enumerate(gestures):
    gesture_path = os.path.join(dataset_directory, gesture)
    for img_name in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_name)
        image = cv2.imread(img_path)
        landmarks = extract_landmarks(image)
        if landmarks:
            landmarks_data.append(landmarks)
            labels.append(gesture_index )

# Save the processed data
np.save('landmarks_data.npy', landmarks_data)
np.save('labels.npy', labels)