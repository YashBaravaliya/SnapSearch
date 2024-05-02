import streamlit as st
import cv2
import os
import numpy as np
import time
import dlib

class App:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src\Models\shape_predictor_68_face_landmarks.dat")

    def camera_input(self):

        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), channels="RGB")

            return cv2_img

    def capture_image(self, folder_path, name):
        # Open a connection to the webcam (adjust the index based on your system)
        capture = cv2.VideoCapture(0)


        # Create an initial empty frame window
        FRAME_WINDOW = st.image([])

        take_photo = 0
        if st.button("Take Photo"):
            take_photo = 1

        count = 0
        while True:
            ret, frame = capture.read()
            FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True, caption="Webcam Feed")

            if take_photo == 1:
                print("Taking photo")
                # Capture the image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_rect = self.detector(gray)

                for face_rect in faces_rect:
                    cropped_face = frame[face_rect.top()-20:face_rect.bottom(), face_rect.left():face_rect.right()]
                    cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                    save_path = os.path.join(folder_path, f"{name}.png")
                    cv2.imwrite(save_path, cropped_face_resized)
                    count += 1

                if count == 1:
                    break

        # Release the webcam and close all windows
        capture.release()
        cv2.destroyAllWindows()

    def submit_attendance(self, name):
        folder_path = os.path.join("data", name)
        os.makedirs(folder_path, exist_ok=True)
        self.capture_image(folder_path, name)
        st.write("Capture Complete")

if __name__ == "__main__":
    st.title("Selfie Capture")
    name = st.text_input("Enter your name:")
    app = App()

    if st.button("Take Selfie"):
        app.submit_attendance(name)
