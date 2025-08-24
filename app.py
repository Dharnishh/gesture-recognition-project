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
