import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from typing import Union, Sequence
import random
from torchvision import transforms
from torchvision.transforms import functional

def random_augmented_image(
    image:    Image,
    image_size: Union[int, Sequence[int]],
    seed:    int
) -> torch.Tensor:
    torch.random.manual_seed(seed)
    image = torchvision.transforms.Resize(image_size)(image)
    options = [
        torchvision.transforms.RandomRotation(180),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=1.3, saturation=0.5, hue=0.5),
    ]
    chosen = random.choices(options, k=2)
    for t in chosen:
        image = t(image)
    transform = torchvision.transforms.ToTensor()(image)
    transform = torch.nn.Dropout(p=0.15)(transform)
    return transform

if __name__ == "__main__":
    with Image.open("08_example_image.jpg") as image:
        transformed_image = random_augmented_image(image, image_size=300, seed=3)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_image))
        axes[1].set_title("Transformed image")
        fig.tight_layout()
        plt.show()




