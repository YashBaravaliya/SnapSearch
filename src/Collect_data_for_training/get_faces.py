import cv2
import os
import numpy as np
from datetime import datetime
import time
import dlib

class App:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src\Models\shape_predictor_68_face_landmarks.dat")

        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.update_camera()

    def submit_attendance(self, name):
        count = 0
        folder_path = os.path.join("data", name)
        os.makedirs(folder_path, exist_ok=True)

        while count < 50:
            ret, frame = self.capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = self.detector(gray)

            for face_rect in faces_rect:
                landmarks = self.predictor(gray, face_rect)
                landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

                cv2.rectangle(frame, (face_rect.left()-5, face_rect.top()-25), (face_rect.right()+5, face_rect.bottom()+5), (0, 255, 0), 2)
                cropped_face = frame[face_rect.top()-20:face_rect.bottom(), face_rect.left():face_rect.right()]
                cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                save_path = os.path.join(folder_path, f"{name}_{count+1}.png")
                cv2.imwrite(save_path, cropped_face_resized)

                for point in landmarks:
                    cv2.circle(frame, tuple(point), 2, (0, 155, 255), 2)

                count += 1

            cv2.imshow('Camera', frame)
            cv2.waitKey(10)
            time.sleep(0.1)

        cv2.destroyAllWindows()
        print("Capture Complete")


    def update_camera(self):
        ret, frame = self.capture.read()

        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Camera', frame)
            cv2.waitKey(10)
            self.update_camera()

if __name__ == "__main__":
    name = input("Enter your name: ")
    app = App()
    app.submit_attendance(name)
