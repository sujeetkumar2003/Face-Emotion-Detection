import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
model = load_model("C:/Users/vishw/OneDrive/Desktop/Coding/New_folder/Emotion Detection/emotion_model.h5", compile=False)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start Video Capture
cap = cv2.VideoCapture(0)  # 0 for webcam

while True:
    # Read Frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw Rectangle Around Face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess Face for Emotion Detection
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))  # Resize to match model input
        face = face / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis= -1)  # Add batch and channel dimensions
        face = np.expand_dims(face, axis= 0)

        # Predict Emotion
        emotion_probabilities = model.predict(face)
        emotion_label = emotion_labels[np.argmax(emotion_probabilities)]

        # Display Emotion on Frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show Video Feed with Annotations
    cv2.imshow('Emotion Detection', frame)

    # Exit on 'q' Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
