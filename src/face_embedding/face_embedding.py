import dlib
import numpy as np
import cv2
import os
import pickle
from imutils import paths

class FaceEmbeddingGenerator:
    def __init__(self, args):
        self.args = args
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src\Models\shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("src\Models\dlib_face_recognition_resnet_model_v1.dat")

    def extract_face_embedding(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        face = faces[0]
        shape = self.predictor(gray, face)

        # Compute a 128-dimension face descriptor
        face_descriptor = self.face_rec_model.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)

    def generate_face_embedding(self):
        print("[INFO] Quantifying faces...")
        image_paths = list(paths.list_images(self.args["data"]))

        known_embeddings = []
        known_names = []
        total = 0

        for (i, image_path) in enumerate(image_paths):
            try:
                print("[INFO] Processing image {}/{}".format(i + 1, len(image_paths)))
                name = image_path.split(os.path.sep)[-2]
                image = cv2.imread(image_path)

                if image is None:
                    print("[WARN] Unable to read image:", image_path)
                    continue

                embedding = self.extract_face_embedding(image)
                if embedding is not None:
                    known_embeddings.append(embedding)
                    known_names.append(name)
                    total += 1
            except Exception as e:
                print("[WARN] Error processing image:", image_path)
                print(e)

        print(total, "faces embedded")

        data = {"embeddings": known_embeddings, "names": known_names}
        with open(self.args["embeddings"], "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    args = {
        "data": "data",
        "embeddings": "src/models/embeddings.pickle"
    }
    face_embedding = FaceEmbeddingGenerator(args)
    face_embedding.generate_face_embedding()
    print("Embedding generated successfully.")
