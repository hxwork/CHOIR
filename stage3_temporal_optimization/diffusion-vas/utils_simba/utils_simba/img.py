import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import io

# image shape should be [height, width, channels] and in range [0, 1]
def show_img(img, normalize=False, title=None, off_axis=True):
    rgb_array = img
    if type(img) == torch.Tensor:
        rgb_tensor = img.detach().cpu()
        rgb_array = rgb_tensor.numpy()
        
    plt.imshow(rgb_array)
    if title:
        plt.title(title)
    if off_axis:
        plt.axis('off')  # Turn off axis labels
        plt.gca().set_axis_off()  # Turn off the axis
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Remove padding
        plt.margins(0, 0)  # Set margins to zero
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove x-axis ticks
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove y-axis ticks
    plt.show()


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image

def read_mask(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        mask = image[:, :, 3]
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask = binary_mask / 255
    else:
        mask = np.ones_like(image[:, :, 0])
    return mask

def compress_image(rgb_array, format="JPEG", quality=75):
    pil_image = Image.fromarray(rgb_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()