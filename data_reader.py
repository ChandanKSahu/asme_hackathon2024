import zipfile
import os

from PIL import Image
import io
import numpy as np

import random
import pickle

import matplotlib.pyplot as plt
import matplotlib
# import PyQt5

# matplotlib.use("Qt5Agg")
import cv2

#%% File locations

zip_loc = r"E:\Hackathon2024Data\NIST\Pretraining\In-situ Meas Data.zip"
root_folder = r"In-situ Meas Data/Melt Pool Camera"


#%% Open and read zipped folder

image_dict ={}

img_avg_dict = {}

with zipfile.ZipFile(zip_loc, 'r') as zip_ref:
    ### list all entries in the xip file
    all_entries = zip_ref.namelist()

    image_files = [f for f in all_entries if f.startswith(root_folder) and f.lower().endswith(".bmp")]

    ## read and process each image
    for im_file in image_files:
        with zip_ref.open(im_file) as file:
            ### read the image into PIL image and conver it to grayscale
            image = Image.open(io.BytesIO(file.read())).convert('L')

            ### convert image to numpy array
            image_array = np.array(image)
            image_array = image_array[4:-4, :]
            image_array = cv2.resize(image_array, (0, 0), fx=0.5, fy=0.5)

            ### generate the key based on file name
            ### extract "L0001" and 'frame00001' from "MIA_L0001/frame00001.bmp"
            parts = im_file.split("/")
            layer = parts[-2].split("_")[-1] ### extract 'L0001'
            frame = parts[-1].split(".")[0] ### extract 'frame00001'
            key = f"{layer}_f{frame[5:]}"

            if np.mean(image_array) > 6:
                image_dict[key] = image_array
                img_avg_dict[key] = np.mean(image_array)

image_dict = dict(sorted(image_dict.items()))

#%% Image filtering

img_avg_dict = {k:np.mean(v) for k,v in image_dict.items()}

plt.figure(figsize=(20, 6))
plt.hist(img_avg_dict.values(), bins=256, density=True, color='gray', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xticks(np.arange(0, 55, 2))
plt.grid(True)
plt.show()

#%% Save data

with open("image_array_dict.pkl", "wb") as file:
    pickle.dump(image_dict, file, pickle.HIGHEST_PROTOCOL)

#%% Randomly Verifying 25 images

selected_keys = random.sample(list(image_dict.keys()), 100)

# Create the list of images based on the selected keys
selected_images = [image_dict[key] for key in selected_keys]


# Function to plot 25 images in a 5x5 grid
def plot_images_grid(images):
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Turn off the axis

    plt.show()


# Plot the randomly selected images in a 5x5 grid
plot_images_grid(selected_images)