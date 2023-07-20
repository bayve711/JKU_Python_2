import numpy as np
from PIL import Image
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

            c_linear = np.where(normalized <= 0.04045, normalized / 12.92,
                                np.power((normalized + 0.055) / 1.055, 2.4))
            y_linear = c_linear[:, :, 0] * 0.2126 + c_linear[:, :, 1] * 0.7152 + c_linear[:, :, 2] * 0.0722
            y = np.where(y_linear <= 0.0031308, y_linear * 12.92,
                         (1.055 * np.power(y_linear, 1 / 2.4)) - 0.055)
            y = y[:, :, np.newaxis]
            output = (y * 255).clip(0, 255)

    else:
        raise ValueError

    if np.issubdtype(pil_image.dtype, np.integer):
        output = np.round(output).astype(pil_image.dtype)

    return output

def round_array(arr: np.ndarray) -> np.ndarray:
    return np.round(np.mean(arr)).astype('int32')


def plot_images(original_image, pixelated_image, known_array, target_array):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(pixelated_image.squeeze(), cmap='gray')
    axes[1].set_title('Pixelated Image')
    axes[1].axis('off')

    axes[2].imshow(known_array.squeeze(), cmap='gray')
    axes[2].set_title('Known Array')
    axes[2].axis('off')

    axes[3].imshow(target_array.squeeze(), cmap='gray')
    axes[3].set_title('Target Array')
    axes[3].axis('off')

    plt.show()


def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    if image.shape[2] != 1:
        raise ValueError
    elif (width and height and size) < 2:
        raise ValueError
    elif x < 0 or x + width > image.shape[0]:
        raise ValueError(x)
    else:
        pixelated_image, known_array, target_array = image.copy(), np.ones_like(image, dtype=bool), image[y:y + height,
                                                                                                    x:x + width]

        h_blocks = height // size + (height % size != 0)
        w_blocks = width // size + (width % size != 0)

        for i in range(h_blocks):
            for j in range(w_blocks):
                y_start = y + i * size
                y_end = min(y + (i + 1) * size, y + height)
                x_start = x + j * size
                x_end = min(x + (j + 1) * size, x + width)


                pixelated_image[y_start:y_end, x_start:x_end] = round_array(
                    pixelated_image[y_start:y_end, x_start:x_end])


                known_array[y_start:y_end, x_start:x_end] = False

        return np.round(pixelated_image).astype(image.dtype), known_array, target_array



# image_files = sorted(glob.glob(os.path.join(input, "**", "*.jpg"), recursive=True))
# print("Filename: ",os.path.basename(image_files[0]))
# with Image.open(image_files[0]) as im:
#     image = np.array(im)
#
# grayscaled_img = to_grayscale(image)
# prepared_img = prepare_image(grayscaled_img, 0, 0, 64, 64, 10)
# plot_images(grayscaled_img, prepared_img[0], prepared_img[1], prepared_img[2])
