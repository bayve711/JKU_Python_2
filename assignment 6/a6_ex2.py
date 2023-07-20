import glob, os
import torch
import torchvision
from PIL import Image
from typing import Union, Sequence
import random
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import Dataset
from a6_ex1 import random_augmented_image
class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        return (image, index)

    def __len__(self) -> int:
        return len(self.image_paths)

class TransformedImageDataset(Dataset):
    def __init__(self, dataset: ImageDataset, image_size: Union[int, Sequence[int]]):
        self.dataset = dataset
        self.image_size = image_size
    def __getitem__(self, index: int):
        img, index = self.dataset[index]
        transformed_img = random_augmented_image(img, self.image_size, seed=index)
        return (transformed_img, index)

    def __len__(self):
        return len(self.dataset)

# if    __name__    ==    "__main__":
#     from matplotlib import pyplot as plt
#
#     imgs = ImageDataset(image_dir="images")
#     transformed_imgs = TransformedImageDataset(imgs, image_size=300)
#     for (original_img, index), (transformed_img, _) in zip(imgs, transformed_imgs):
#         fig, axes = plt.subplots(1, 2)
#         axes[0].imshow(original_img)
#         axes[0].set_title("Original image")
#         axes[1].imshow(transforms.functional.to_pil_image(transformed_img))
#         axes[1].set_title("Transformed image")
#         fig.suptitle(f"Image {index}")
#         fig.tight_layout()
#         plt.show()
