import streamlit as st
from research.prediction import FaceRecognizer
import os
import cv2
import numpy as np
from PIL import Image

def predict_faces_in_folder(folder_path, user_img):
    # Load the face recognizer
    face_recognizer = FaceRecognizer()

    # Output folder path to save personalized images
    output_folder = "personalized_images"

    temp = 0
    cols = st.columns(4)

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform face recognition
            predicted_label = face_recognizer.recognize_faces(image)

            # Save personalized image with original file name
            # Ask user for label and save the image if it contains a face
            if predicted_label != "No face detected" and predicted_label == user_img:
                face_recognizer.save_image(image, filename, output_folder)

                # Display the uploaded image and prediction dynamically
                cols[temp % 4].image(image, caption=f"Predicted Label: {predicted_label}", use_column_width=True)
                temp += 1

if __name__ == "__main__":
    st.set_page_config(page_title="SnapSearch - AI Photo Search", page_icon=":camera:")

    st.title("SnapSearch - AI Photo Search")

    st.markdown(
        "SnapSearch is an AI-powered photo search tool that allows users to quickly and effortlessly find photos of themselves "
        "from a vast collection of images. Upload your photos and let SnapSearch do the rest!"
    )

    folder_path = st.text_input("Enter the path to the folder containing images:")
    user_img = st.text_input("Enter the name of the person whose image you want to predict:")

    if st.button("Search"):
        if folder_path and user_img:
            predict_faces_in_folder(folder_path, user_img)
        else:
            st.error("Please provide a valid folder path and user image name.")