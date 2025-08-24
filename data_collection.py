import cv2
import os

def collect_gesture_data(gesture_name, num_samples=100):
    os.makedirs(gesture_name, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Collecting Data', frame)
            cv2.imwrite(f"{gesture_name}/{gesture_name}_{count}.jpg", frame)
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gestures = ['sit', 'stand', 'fly']
    for gesture in gestures:
        print(f"Collecting data for gesture: {gesture}")
        collect_gesture_data(gesture)