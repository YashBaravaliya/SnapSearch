import cv2
import numpy as np
import pickle
from keras.models import load_model
import dlib

class FaceRecognizer:
    def __init__(self):
        # Load the trained Keras face recognition model
        self.model = load_model("src/models/face_recognition_model.h5")

        # Load the label encoder
        with open("src/models/label_encoder.pickle", "rb") as le_file:
            self.le = pickle.load(le_file)

        # Load the face detection and landmarks detection models from dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src/models/shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("src/models/dlib_face_recognition_resnet_model_v1.dat")

    def preprocess_input(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        # Extract face embeddings
        face = faces[0]
        shape = self.predictor(gray, face)
        face_descriptor = self.face_rec_model.compute_face_descriptor(image, shape)

        # Convert the face descriptor to numpy array
        embedding = np.array(face_descriptor)
        return embedding

    def recognize_faces(self, image):
        # Preprocess the input image
        embedding = self.preprocess_input(image)

        if embedding is not None:
            # Reshape the embedding to match the input shape expected by the model
            embedding = embedding.reshape(1, -1)

            # Predict using the pre-trained Keras model
            preds = self.model.predict(embedding)
            preds = preds.flatten()

            # Decode the predicted label using the label encoder
            predicted_label = self.le.inverse_transform([np.argmax(preds)])[0]
            return predicted_label
        else:
            return "No face detected"

if __name__ == "__main__":
    # Load the face recognizer
    face_recognizer = FaceRecognizer()

    # Load the image
    image_path = "20240118084646_IMG_1030.JPG"
    image = cv2.imread(image_path)

    # Perform face recognition
    predicted_label = face_recognizer.recognize_faces(image)
    print("Predicted Label:", predicted_label)