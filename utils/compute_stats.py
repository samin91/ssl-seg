import os
import numpy as np
from PIL import Image


def compute_dataset_stats(images_dir, split="train"):
    train_images = images_dir + "/" + split + "/images"
    imgs = [
        np.array(
            Image.open(os.path.join(train_images, f)).convert("L"), dtype=np.float32
        )
        / 255.0
        for f in os.listdir(train_images)
    ]
    imgs = np.stack(imgs, axis=0)
    mean = imgs.mean()
    std = imgs.std()
    return mean, std
