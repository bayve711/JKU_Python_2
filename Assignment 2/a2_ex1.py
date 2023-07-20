from PIL import Image
import numpy as np
import glob, os
import matplotlib.pyplot as plt

input = "/Users/bayve/Desktop/JKU Python 2/Assignment 2/04_images"

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if len(pil_image.shape) == 2:
        output = pil_image[np.newaxis, :, :].copy()
    elif len(pil_image.shape) == 3:
        if pil_image.shape[2] != 3:
            raise ValueError
        else:
            normalized = pil_image / 255
            c_linear = np.where(normalized <= 0.4045, normalized/12.92, ((normalized + 0.055)/1.055)**2.4)
            y_linear = 0.2126*c_linear[:, :, 0] +0.7152*c_linear[:, :, 1] + 0.0722*c_linear[:, :, 2]
            pil_y = np.where(y_linear <= 0.0031308, 12.92 * y_linear, 1.055 * (y_linear) ** (1/2.4) - 0.055)
            output = (pil_y * 255).clip(0, 255)
    else:
        raise ValueError
    if np.issubdtype(pil_image.dtype, np.integer):
        output = np.round(output).astype(pil_image.dtype)

    return output


# image_files = sorted(glob.glob(os.path.join(input, "**", "*.jpg"), recursive=True))
# print("Filename: ",os.path.basename(image_files[0]))
# with Image.open(image_files[0]) as im:  # This returns a PIL image
#     image = np.array(im)  # We can convert it to a numpy array
#     print("SHAPE:", image.shape)
# print("image data:")
# print(f"mode: {im.mode}; shape: {image.shape}; min: {image.min()}; max: {image.max()}; dtype: {image.dtype}")
#
# grayscaled_img = to_grayscale(image)
#
# plt.imshow(grayscaled_img)
# plt.show()

