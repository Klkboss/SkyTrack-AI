import torch
import torchvision
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
model = model.to(device)
model.eval()

def get_road_mask(frame):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(512,512))

    img = img / 255.0
    img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)["out"][0]

    output = output.argmax(0).cpu().numpy()

    # road class
    mask = (output == 7).astype("uint8")

    mask = cv2.resize(mask,(frame.shape[1],frame.shape[0]))

    return mask