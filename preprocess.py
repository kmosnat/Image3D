# preprocess.py
# shi zhang
# This file resize the image for uniformity
# and apply filters

# import statements
import os
import cv2
import numpy as np


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    return gray_image


def process_images_in_folder(folder_path, output_folder=None):
    print(f"Processing images in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path}...")
            image = cv2.imread(file_path)
            processed_image = preprocess_image(image)

            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Saved processed image to {output_path}")