import os
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split

"""
Function reads in images from a given directory, normalizes them, and splits them 
into L, a, b channels. The L channel is appended to the input of the dataset and 
The a and b channels are merged and appended to the targets of the dataset. The
dataset is then split into train/test sets and then used to create a tf.data.Dataset
object. This Dataset is batched and returned.
"""
def prepare_dataset(data_dir, batch_size=32, dataset_split=2500):
    x = []  # Inputs
    y = []  # Targets
    for image_file in os.listdir(data_dir)[0: dataset_split]:
        rgb_image = Image.open(os.path.join(data_dir, image_file)).resize((128, 128))
        # Normalize RGB image array
        rgb_image_arr = (np.asarray(rgb_image)) / 255
        if len(rgb_image_arr.shape) == 3:  # Simple validation to make sure images have three color channels
            # Convert to LAB
            lab_image_arr = rgb2lab(rgb_image_arr)
            # Normalize LAB image array
            lab_image_arr = (lab_image_arr + [0, 128, 128]) / [100, 255, 255]
            # Split into l and ab channels
            l, a, b = np.split(lab_image_arr, 3, axis=-1)
            ab = np.concatenate((a, b), axis=-1)
            # Append images to arrays 
            x.append(l)
            y.append(ab)

    # Split array into train/test sets
    train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.1)
    # Construct Dataset object 
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset = dataset.batch(batch_size)
    return dataset
