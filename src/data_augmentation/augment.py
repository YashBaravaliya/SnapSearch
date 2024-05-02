import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment_images(data_folder, augmentations, max_images=100):
    """
    Augments images in the given data folder using the specified augmentations.

    Parameters:
        data_folder (str): Path to the data folder containing images.
        augmentations (ImageDataGenerator): Keras ImageDataGenerator object for augmentation.
        max_images (int): Maximum number of augmented images to generate for each original image. Default is 100.

    Returns:
        None
    """
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' does not exist.")
        return

    print("Augmenting images in folder:", data_folder)

    # Instantiate ImageDataGenerator for augmentation
    datagen = augmentations

    # Iterate over each image in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(data_folder, filename)
            input_img = cv2.imread(image_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            # Reshape image to (1, height, width, channels) for Keras generator
            input_img = input_img.reshape((1,) + input_img.shape)
            # Generate augmented images
            count = 0
            for batch in datagen.flow(input_img, batch_size=1):
                augmented_image = batch[0].astype(np.uint8)
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(data_folder, f"aug_{filename}_{count}.jpg"), augmented_image)
                count += 1
                if count >= max_images:
                    break

    print("Augmentation completed for folder:", data_folder)



# Define Keras augmentation parameters
augmentations = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

if __name__ == "__main__":
    # Path to your data folder
    data_folder = "temp"

    # Call the function to augment images
    augment_images(data_folder, augmentations, max_images=100)
