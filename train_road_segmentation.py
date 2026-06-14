import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda"

dataset_path = r"C:\DroneProject\Datasets\SkyScapes\Segmentation"

class RoadDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform = A.Compose([
            A.Resize(512,512),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_name = os.path.splitext(self.images[index])[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = mask / 255.0
        if mask is None:
            raise ValueError(f"Mask not found: {mask_path}")

        augmented = self.transform(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"].unsqueeze(0)

        return image, mask


train_dataset = RoadDataset(
    dataset_path + "/train/images",
    dataset_path + "/train/masks"
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

model = model.to(device)

loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 20

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss:", total_loss)

torch.save(model.state_dict(),"road_segmentation_model.pth")

print("Training finished")