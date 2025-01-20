# Real-time Emotion Recognition

This project implements real-time emotion recognition using OpenCV and TensorFlow. The system can detect faces and classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Structure

- `emotion_recognition.py`: Main script for real-time emotion detection
- `train_model.py`: Script for training the emotion recognition model
- `requirements.txt`: List of required Python packages
- `best_model.h5`: Pre-trained model
- `data/`: Directory containing the FER2013 dataset

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Training the Model (optional, if you want to retrain):
```bash
python train_model.py
```

2. Running Emotion Recognition:
```bash
python emotion_recognition.py
```
Press 'q' to quit the application.

## Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional layers for feature extraction
- Batch normalization for training stability
- MaxPooling layers for dimensionality reduction
- Dropout layers to prevent overfitting
- Dense layers for final classification

## Dataset

The model is trained on the FER2013 dataset, which contains facial expressions categorized into seven emotions.
