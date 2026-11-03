
import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models_artifacts",
    "final_emotion_model.keras"
)

IMG_SIZE = (48, 48)

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

model = None

# OpenCV's built-in face detector
CASCADE_PATH = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def get_model():
    global model

    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    return model


def detect_face(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        raise ValueError("No face detected. Please face the camera.")

    # Select the largest detected face
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    return gray[y:y+h, x:x+w]


def preprocess_face(face_img):
    face = cv2.resize(face_img, IMG_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    return face


def predict_emotion(image):
    face = detect_face(image)

    model = get_model()
    processed = preprocess_face(face)

    predictions = model.predict(processed, verbose=0)

    class_idx = np.argmax(predictions)
    emotion = CLASS_NAMES[class_idx]
    confidence = float(np.max(predictions))

    return {
        "emotion": emotion,
        "confidence": round(confidence, 4)
    }
