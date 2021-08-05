import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import matplotlib.pyplot as plt

# Util function to convert image from RGB to LAB and normalize
def rgb_to_lab_norm(image):
    image = normalize(image, rgb=True)
    image = normalize(image, rgb=False)
    return image

# Util function to normalize image. Image is normalized differently depending on colorspace.
def normalize(image, rgb=True):
    if rgb:
        image = np.asarray(image) / 255.0
    else:
        image = np.asarray(image) 
        image = (image + [0, 128, 128]) / [100, 255, 255]
    return image

# Util function to normalize a single channel RGB image
def normalize_one_channel(image):
    image = np.asarray(image) / 255.0
    # image = image / 100
    return image

# Util function to reverse normalization of an image
def unnormalize_lab(image):
    image = (image * [100, 255, 255]) - [0, 128, 128]
    image = lab2rgb(image)
    image = image * 255
    return image

# Util function to merge L and AB images into one LAB image
def merge_images(l, ab):
    merged_image = np.zeros((128, 128, 3))
    merged_image[:, :, 0] = l[:, :, 0]  # or L[0][:, :, 0]
    merged_image[:, :, 1:] = ab[0]
    return merged_image

"""
Util function to perform the necessary merging and conversions to convert the output
of the generator to a conventional RGB image
"""
def interpret_output(merged_image):
    output_image = (merged_image * [100, 255, 255]) - [0, 128, 128]
    output_image = lab2rgb(output_image)
    output_image = output_image * 255
    output_image = output_image.astype('uint8')
    return output_image

"""
Util function to display the output of the generator. Used for debugging and running
the program from an IDE
"""
def show_output(output_image):
    output_image = Image.fromarray(output_image).resize((1024, 1024))
    output_image = np.asarray(output_image)
    plt.imshow(output_image)
    plt.show()
