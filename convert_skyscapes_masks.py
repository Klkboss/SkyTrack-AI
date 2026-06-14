import os
import cv2
import numpy as np

dataset = r"C:\DroneProject\Datasets\SkyScapes\SS_Dense"

input_labels = os.path.join(dataset, "train", "labels", "grayscale")
output_masks = os.path.join(dataset, "train", "road_masks")

os.makedirs(output_masks, exist_ok=True)

ROAD_CLASS_ID = 1   # road class in grayscale labels

for file in os.listdir(input_labels):

    if not file.endswith(".png"):
        continue

    path = os.path.join(input_labels, file)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    binary = np.zeros_like(mask)

    binary[mask == ROAD_CLASS_ID] = 255

    cv2.imwrite(os.path.join(output_masks, file), binary)

print("Road mask conversion finished.")
#this script converts:
# road = white
# background = black