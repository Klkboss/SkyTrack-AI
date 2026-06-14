import os
import shutil

# VisDrone YOLO dataset
vis_img = r"C:\DroneProject\Datasets\VISDRONE\yolo\images"
vis_lbl = r"C:\DroneProject\Datasets\VISDRONE\yolo\labels"

# UAVDT dataset
uav_train_img = r"C:\DroneProject\Datasets\UAVDT\train\images"
uav_train_lbl = r"C:\DroneProject\Datasets\UAVDT\train\labels"

uav_val_img = r"C:\DroneProject\Datasets\UAVDT\val\images"
uav_val_lbl = r"C:\DroneProject\Datasets\UAVDT\val\labels"

# Output dataset
train_img = r"C:\DroneProject\Datasets\DroneCombined\train\images"
train_lbl = r"C:\DroneProject\Datasets\DroneCombined\train\labels"

val_img = r"C:\DroneProject\Datasets\DroneCombined\val\images"
val_lbl = r"C:\DroneProject\Datasets\DroneCombined\val\labels"


def copy_dataset(img_src, lbl_src, img_dst, lbl_dst, prefix):

    for file in os.listdir(img_src):

        new_name = prefix + "_" + file

        shutil.copy2(
            os.path.join(img_src, file),
            os.path.join(img_dst, new_name)
        )

        label = file.replace(".jpg", ".txt")

        label_src = os.path.join(lbl_src, label)

        if os.path.exists(label_src):

            shutil.copy2(
                label_src,
                os.path.join(lbl_dst, new_name.replace(".jpg", ".txt"))
            )


print("Copying VisDrone...")
copy_dataset(vis_img, vis_lbl, train_img, train_lbl, "vis")

print("Copying UAVDT train...")
copy_dataset(uav_train_img, uav_train_lbl, train_img, train_lbl, "uav")

print("Copying UAVDT val...")
copy_dataset(uav_val_img, uav_val_lbl, val_img, val_lbl, "uav")

print("Dataset merge complete")