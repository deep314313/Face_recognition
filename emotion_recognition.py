import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque
import os

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face):
    """
    Preprocess face image for the model
    """
    # Resize to 48x48
    face = cv2.resize(face, (48, 48))
    
    # Ensure RGB
    if len(face.shape) == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel values
    face = face.astype('float32') / 255.0
    
    return np.expand_dims(face, axis=0)

def get_emotion_prediction(model, face, emotion_buffer):
    """
    Get emotion prediction with temporal smoothing
    """
    # Get model prediction
    prediction = model.predict(face, verbose=0)[0]
    emotion_buffer.append(prediction)
    
    # Average the predictions for smoothing
    avg_prediction = np.mean(emotion_buffer, axis=0)
    emotion_idx = np.argmax(avg_prediction)
    emotion = emotions[emotion_idx]
    confidence = avg_prediction[emotion_idx]
    
    return emotion, confidence

def main():
    # Load the trained model
    model_path = 'best_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first using train_model.py")
        return
    
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device!")
        return
    
    # Initialize emotion buffer for temporal smoothing
    emotion_buffer = deque(maxlen=5)  # Reduced buffer size for faster response
    
    print("Starting emotion recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face_roi = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_face(face_roi)
            
            # Initialize buffer if needed
            if len(emotion_buffer) < emotion_buffer.maxlen:
                prediction = model.predict(preprocessed_face, verbose=0)[0]
                for _ in range(emotion_buffer.maxlen):
                    emotion_buffer.append(prediction)
            
            # Get emotion prediction
            emotion, confidence = get_emotion_prediction(model, preprocessed_face, emotion_buffer)
            
            # Draw rectangle and emotion label
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add emotion label with confidence
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Emotion Recognition', frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
