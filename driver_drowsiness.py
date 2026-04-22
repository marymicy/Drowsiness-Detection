# Importing required libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils

# Initializing the camera
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status markers
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    """Determine eye blinking status based on eye aspect ratio."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Eyes Open
    elif 0.21 <= ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping

while True:
    ret, frame = cap.read()
    
    # Ensure frame is captured properly
    if not ret or frame is None:
        print("Error: Could not read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    # Initialize face_frame to prevent "name not defined" errors
    face_frame = frame.copy()  # Default to full frame
    
    # If no faces are detected, use a blank image for 'face_frame'
    if len(faces) == 0:
        face_frame = np.zeros_like(frame)

    # Process each detected face
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Check blinking for both eyes
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Update drowsiness state
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            sleep = 0
            drowsy = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display status text
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw facial landmark points
        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Display the processed frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)

    # Press 'Esc' key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
