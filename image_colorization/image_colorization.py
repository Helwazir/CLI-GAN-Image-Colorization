import os, sys
from utils import image_utils as image_utils
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import argparse
import numpy as np
import glob
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')

# Create parser for command line arguments
parser = argparse.ArgumentParser(description='Colorize black & white images in a folder')
parser.add_argument('-p',
                    '--path',
                    dest='path',
                    type=str,
                    default='test_input',
                    help='the path to the folder containing the images to colorize')
parser.add_argument('-sd',
                    '--savedir',
                    dest='savedir',
                    type=str,
                    default='test_output',
                    help='the path to the folder to save the colorized images to')
parser.add_argument('-g',
                    '--generator',
                    dest='generator',
                    type=int,
                    choices=[0, 1],
                    default=0,
                    help='Which generator to use, 0 for the generator architecture I wrote, 1 for the proven architecture')
args = parser.parse_args()

# Parse args for values
colorize_dir = args.path
save_dir = args.savedir
args.generator

# Load previously trained generator
# colorizer = load_model('./model')
if args.generator == 0:
    # Load my generator
    colorizer = load_model('generators/generator_0')
elif args.generator == 1:
    print('there')
    # Load the proven generator
    colorizer = load_model('generators/generator_1')

# Get the paths of the images to colorize
extensions = ('/*.png', '/*.jpeg', '/*.jpg', '/*.tiff')
image_files = []
for ext in extensions:
    image_files.extend([os.path.basename(x) for x in glob.glob(colorize_dir + ext)])

"""
Read the images in. If an image has more than one color channel
convert it to one channel. Normalize the images between 0 and 1.
"""
inputs = []
for image_file in image_files:
    image = Image.open(os.path.join(colorize_dir, image_file)).resize((128, 128))
    if image.mode != 'L':
        image = image.convert('L')
    image = np.asarray(image).astype('uint8')

    image = image.reshape(128, 128, 1)
    image = image_utils.normalize_one_channel(image)
    inputs.append(image)

# Create Dataset from images and batch it with a batch size of 1
dataset = tf.data.Dataset.from_tensor_slices(inputs)
dataset = dataset.batch(1)

"""
Feed each image to the colorizer. Merge the resulting color channels
with the input image to create a 3 channel image. Convert the image
to RGB and de-normalize the pixel values.
"""
i = 0
for image in tqdm(dataset):
    # Colorize image
    y = colorizer(image).numpy()
    # Merge L and ab color channels
    merged_image = image_utils.merge_images(inputs[i], y)
    # Perform necessary conversions to un-normalize the pixel values
    output_image = image_utils.interpret_output(merged_image)
    # Save the colorized image
    save_image = Image.fromarray(output_image)
    save_image.save(os.path.join(save_dir, 'colorized_' + image_files[i]))
    i += 1
