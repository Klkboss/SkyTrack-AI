import os
from PIL import Image

dataset_path = r"C:\DroneProject\Datasets\DroneCombined"

splits = ["train", "val"]

total_images = 0
missing_labels = 0
empty_labels = 0
bad_labels = 0
broken_images = 0

for split in splits:
    img_dir = os.path.join(dataset_path, split, "images")
    lbl_dir = os.path.join(dataset_path, split, "labels")

    for img_name in os.listdir(img_dir):
        if not img_name.endswith(".jpg"):
            continue

        total_images += 1
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))

        # check image corruption
        try:
            Image.open(img_path).verify()
        except:
            broken_images += 1
            print("Broken image:", img_path)

        # check missing label
        if not os.path.exists(label_path):
            missing_labels += 1
            print("Missing label:", label_path)
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        # check empty label
        if len(lines) == 0:
            empty_labels += 1
            continue

        # check label format
        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                bad_labels += 1
                print("Bad label format:", label_path)
                break

            class_id = int(parts[0])

            if class_id < 0:
                bad_labels += 1
                print("Invalid class id:", label_path)
                break

print("\nDATASET REPORT")
print("Total images:", total_images)
print("Missing labels:", missing_labels)
print("Empty labels:", empty_labels)
print("Bad labels:", bad_labels)
print("Broken images:", broken_images)