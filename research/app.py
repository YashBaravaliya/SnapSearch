import streamlit as st
from research.prediction import FaceRecognizer
import cv2
import numpy as np

def predict_face(image):
    # Load the face recognizer
    face_recognizer = FaceRecognizer()

    # Perform face recognition
    predicted_label = face_recognizer.recognize_faces(image)
    return predicted_label

if __name__ == "__main__":
    st.title("Face Recognition Predictor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Perform prediction
        predicted_label = predict_face(image)

        # Display prediction
        st.write(f"Predicted Label: {predicted_label}")
