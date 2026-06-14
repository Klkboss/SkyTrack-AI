import json

path = r"C:\DroneProject\Datasets\VISDRONE\coco\VisDrone2019-MOT-train.json"

data = json.load(open(path))

print("Images:", len(data["images"]))
print("Annotations:", len(data["annotations"]))
print("Categories:", data["categories"])

# sample annotation
print("\nSample annotation:")
print(data["annotations"][0])

bad = 0

for ann in data["annotations"]:
    x,y,w,h = ann["bbox"]

    if w <= 0 or h <= 0:
        bad += 1

print("Invalid boxes:", bad)