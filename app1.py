import streamlit as st
import cv2
import os
import numpy as np
import time
import dlib
from src.data_augmentation.augment import augment_images,augmentations
from src.face_embedding.face_embedding import FaceEmbeddingGenerator
from src.traning.train import *

class App:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src\Models\shape_predictor_68_face_landmarks.dat")

    def camera_input(self,name):

        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces_rect = self.detector(gray)

            for face_rect in faces_rect:
                cropped_face = cv2_img[face_rect.top()-70:face_rect.bottom()+70, face_rect.left()-70:face_rect.right()+70]
                cropped_face_resized = cv2.resize(cropped_face, (112, 112))
    
            st.image(cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB), channels="RGB")
            folder_path = os.path.join("data", name)
            if os.path.exists(folder_path):
                # Delete all files inside the directory
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    os.remove(file_path)
            else:
                os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(os.path.join(folder_path, f"{name}.png"), cropped_face_resized)
            st.success("Image saved successfully")

            return cv2_img
        
    
if __name__ == "__main__":
    st.title("Selfie Capture")
    name = st.text_input("Enter your name:")
    data_folder = os.path.join("data", name)
    print(data_folder)
    app = App()
    captured_image = app.camera_input(name)  # Capture the image
    args_embdding = {
        "data": "data",
        "embeddings": "src/models/embeddings.pickle"
    }
    if captured_image is not None:
        if st.button("Train Data"):
            augment_images(data_folder, augmentations, max_images=50)
            st.success("Data augmentation completed successfully")
            face_embedding = FaceEmbeddingGenerator(args_embdding)
            face_embedding.generate_face_embedding()
            st.success("Face embeddings generated successfully")
            data_folder_path = "data"
            arguments = {
                "model": "src/models/face_recognition_model.h5",
                "le": "src/models/label_encoder.pickle"
            }

            # Load the face recognition model from dlib
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("src/models/shape_predictor_68_face_landmarks.dat")
            face_rec_model = dlib.face_recognition_model_v1("src/models/dlib_face_recognition_resnet_model_v1.dat")

            embeddings = generate_face_embedding(data_folder_path, detector, predictor, face_rec_model)
            train_model(embeddings, arguments)
            st.success("Model trained successfully")
            
