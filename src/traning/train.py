import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from architecture import Architecture
from keras.callbacks import Callback, EarlyStopping
import keras
import dlib
import cv2

class ProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.epochs}")
        print(f"Loss: {logs['loss']}, Acc: {logs['accuracy']}")

def extract_face_embedding(image, detector, predictor, face_rec_model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    shape = predictor(gray, face)

    # Compute a 128-dimension face descriptor
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

def generate_face_embedding(data_folder_path, detector, predictor, face_rec_model):
    print("[INFO] Quantifying faces...")
    image_paths = [os.path.join(data_folder_path, folder) for folder in os.listdir(data_folder_path)]

    known_embeddings = []
    known_names = []
    total = 0

    for folder in os.listdir(data_folder_path):
        folder_path = os.path.join(data_folder_path, folder)
        for image_file in os.listdir(folder_path):
            print("[INFO] Processing image {}/{}".format(total + 1, len(image_paths)))
            image_path = os.path.join(folder_path, image_file)
            name = folder
            image = cv2.imread(image_path)

            embedding = extract_face_embedding(image, detector, predictor, face_rec_model)
            if embedding is not None:
                known_embeddings.append(embedding)
                known_names.append(name)
                total += 1

    print(total, "faces embedded")

    data = {"embeddings": np.array(known_embeddings), "names": known_names}
    return data

def train_model(embeddings, arguments):
    data = embeddings
    labels = data["names"]
    embeddings = data["embeddings"]

    # Use label encoder to convert string labels to numeric values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Define input shape
    input_shape = X_train.shape[1]

    # Build the softmax model
    arc = Architecture(input_shape=(input_shape,), num_classes=num_classes)
    model = arc.build_model()

    # Train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    progress_callback = ProgressCallback(epochs=50)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, callbacks=[progress_callback])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    # Save the trained face recognition model
    model.save(arguments['model'])
    label_encoder_file = open(arguments["le"], "wb")
    label_encoder_file.write(pickle.dumps(label_encoder))
    label_encoder_file.close()
    print("Training Complete. Model trained seamlessly on fresh data.")

if __name__ == "__main__":
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
