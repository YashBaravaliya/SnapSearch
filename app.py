import streamlit as st
import cv2
import os
import numpy as np
import dlib
from src.data_augmentation.augment import augment_images, augmentations
from src.face_embedding.face_embedding import FaceEmbeddingGenerator
from src.traning.train import train_model, generate_face_embedding
from research.prediction import FaceRecognizer

class SnapSearchApp:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def capture_selfie(self, name):
        selfie_buffer = st.camera_input("📸 SnapSearch - AI Photo Search")

        if selfie_buffer is not None:
            bytes_data = selfie_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces_rect = self.face_detector(gray_img)

            for face_rect in faces_rect:
                cropped_face = cv2_img[face_rect.top() - 70:face_rect.bottom() + 70,
                               face_rect.left() - 70:face_rect.right() + 70]
                cropped_face_resized = cv2.resize(cropped_face, (112, 112))

            st.image(cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB), channels="RGB")
            folder_path = os.path.join("data", name)
            if os.path.exists(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    os.remove(file_path)
            else:
                os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(os.path.join(folder_path, f"{name}.png"), cropped_face_resized)
            st.success("🌟 Selfie saved successfully")
            return cv2_img

def predict_faces_in_folder(folder_path, user_img):
    face_recognizer = FaceRecognizer()
    output_folder = "personalized_images"
    temp = 0
    cols = st.columns(4)

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predicted_label = face_recognizer.recognize_faces(image)
            if predicted_label != "No face detected" and predicted_label == user_img:
                face_recognizer.save_image(image, filename, output_folder)
                cols[temp % 4].image(image, caption=f"Predicted Label: {predicted_label}", use_column_width=True)
                temp += 1

if __name__ == "__main__":
    st.set_page_config(page_title="SnapSearch - AI Photo Search", page_icon=":camera:")
    st.title("🔍 SnapSearch: Find Your Selfie")
    st.success(
        "SnapSearch is an AI-powered photo search tool that allows users to quickly and effortlessly find photos of themselves "
        "from a vast collection of images. Upload your photos and let SnapSearch do the rest!"
    )
    folder_path = st.text_input("📂 Enter the path to the folder containing images:")
    name = st.text_input("👤 Enter your name:")
    data_folder = os.path.join("data", name)
    app = SnapSearchApp()
    captured_image = app.capture_selfie(name)
    args_embedding = {
        "data": "data",
        "embeddings": "src/models/embeddings.pickle"
    }
    if captured_image is not None:
        if st.button("🔍 Seek Your Signature Snap"):
            i = 0
            progress_bar = st.progress(i)
            
            augment_images(data_folder, augmentations, max_images=50)
            progress_bar.progress(i + 33)
            face_embedding = FaceEmbeddingGenerator(args_embedding)
            face_embedding.generate_face_embedding()
            progress_bar.progress(i + 66)
            data_folder_path = "data"
            arguments = {
                "model": "src/models/face_recognition_model.h5",
                "le": "src/models/label_encoder.pickle"
            }
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("src/models/shape_predictor_68_face_landmarks.dat")
            face_rec_model = dlib.face_recognition_model_v1("src/models/dlib_face_recognition_resnet_model_v1.dat")
            embeddings = generate_face_embedding(data_folder_path, detector, predictor, face_rec_model)
            train_model(embeddings, arguments)
            progress_bar.progress(100)
            st.balloons()
            if folder_path and name:
                predict_faces_in_folder(folder_path, name)
            else:
                st.error("❌ Please provide a valid folder path and user image name.")
