import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

# Define the augmentation transformations
augmentations = [
    iaa.Fliplr(p=1.0),  # Horizontal Flip
    iaa.Affine(rotate=(10, -10)),  # Rotation
    iaa.GammaContrast((0.4, 0.5)),  # Light Gamma Contrast
    iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6)),  # Sigmoid Contrast
    iaa.LinearContrast((0.6, 0.4))  # Linear Contrast
]

# Path to your data folder
data_folder = "data"

# Iterate over each subfolder (enrollment number)
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)
    if os.path.isdir(folder_path):
        print("Augmenting images in folder:", folder_name)
        # Iterate over each image in the subfolder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                input_img = imageio.imread(image_path)
                # Apply each augmentation transformation to the image
                augmented_images = [aug.augment_image(input_img) for aug in augmentations]
                # Save augmented images with appropriate filenames
                for i, augmented_image in enumerate(augmented_images):
                    output_filename = f"{filename.split('.')[0]}_aug{i}.{filename.split('.')[1]}"
                    output_path = os.path.join(folder_path, output_filename)
                    imageio.imwrite(output_path, augmented_image)
        print("Augmentation completed for folder:", folder_name)
