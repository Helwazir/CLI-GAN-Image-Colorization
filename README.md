# CLI-GAN-Image-Colorization
This is a command line program that allows the user to colorize all the images in a given folder using a U-Net generator model. The models were constructed using the Keras API and the structures of the models were informed by the models described in the sources listed in the resources section. The generators were trained using a dataset of 2500 images, primarily featuring nature scenes, landscaped, and buildings obtained from [1].

## How to Use:
1. Download python 3 on your machine
2. Download the image_colorization folder
3. Navigate into the image_colorization folder in using the command line
4. Run `pip3 install -r requirements.txt`. This will install all necessary dependencies
5. Run `Python3 image_colorization.py -p /path_to_input_dir/ -sd /path to save_dir`. This will colorize all the images in the directory specified by the `-p` argument and save the colorized images to the directory specified by the `-sd` argument. It is important that the full path is entered for both directories, the program will not function correctly otherwise.

### Available Command Line Arguments:
`--path` or `-p` (optional): The complete path to the directory containing the images you wish to colorize. Defaults to an included directory of eight images named "test_input".

`--savedir` or `-sd` (optional): The complete path to the directory to save the colorized images to. Defaults to an included directory named "test_output".

`--generator` or `-g` (0, 1) (optional): Which generator to use to colorize the images. 0 is a generator designed by me, 1 is the generator from [1] modified to work with LAB images. Defaults to 0.


# Model Architectures:
The generator utilizes a U-Net architecture consisting of six encoding blocks and six decoding blocks with an increasing number of filters while encoding and a decreasing number of filters while decoding. Skip connections provide the network with the ability to skip deeper blocks while learning. The structure of each encoding block is made up of two Conv2D layers to downscale the image, a BatchNormalization layer, and a LeakyReLU activation layer. The First Conv2D layer has a stride of one and a dilation rate of four while the second has a stride of two and a dilation rate of zero. Each block has a BatchNormalization layer except for the first one. The LeakyReLU layer has a negative slope of 0.2. Each decoding block is made up of one Conv2DTranspose layer to upscale the image, BatchNormalization, Dropout, Concatenate layer, and relu activation layer. The Conv2Dtranspose layer has a stride of two Dropout is active on all blocks except for the last three. The Concatenate layer connects the decode layer with the mirroring encoding layer to create a skip connection. A final Conv2Dtranspose layer with a dimensionality of two is used at the output with tanh activation.


The architecture of the discriminator closely mirrors the encoding portion of the discriminator. The discriminator is made up of four blocks closely resembling the encoding blocks found in the generator model. Each discriminator block is made up of a Conv2D layer, BatchNormalization, and a LeakyReLU activation layer. BatchNormalization is present in every block except for the first one. A final Conv2D layer with a dimensionality of one is used at the output with no activation function.

# Training and Dataset:
Two generators are available for use in the program. The first generator is one that I designed through trial and error and experimentation and with different aspects of the generators described in [2] [3] [4]. The second generator is a more proven model [1] originally designed to work with RGB images. The code for the second generator is not included in the program folder, only the trained model. The only changes made to the second generator were to modify it to work with LAB images. Both generators were trained against the same discriminator. Both generators and the discriminator were trained using a dataset of 2500 images, primarily made up of nature, landscapes, and buildings, obtained from [1].
Each image was converted from RGB to LAB, split into L and AB images, and both L and AB images were normalized. The generator receives an L image of shape (128, 128, 1) and outputs an AB image of shape (128, 128, 2). The generated AB image is then merged with the input image L, denormalized, and converted back to RGB to receive the final colorized output image. The intuition behind using LAB color space over RGB or BGR is that by giving the generator an L channel image, which looks like a black and white image, it only has to then generate two color channels. Contrast this with using RGB, where lightness is a component of the color channels. To use RGB the generator would receive a black and white image of shape (128, 128, 1), the same shape as the input when using LAB, and be asked to generate an RGB image of shape (128, 128, 3). By using LAB we minimize the number of outcomes that the generator can output. The models were trained for 350 epochs with a batch size of 64. Discriminator loss was calculated using binary cross entropy and generator loss was calculated using mean squared error.

# Libraries and Dependencies:
1. TensorFlow: for use as Keras backend.
2. Keras API: used for constructing and traianing the models.
3. numpy: for image array manipulations.
4. Pillow: for easy reading and converting of image files.
5. scikit-image: for converting from RGB to LAB and from LAB to RGB.
6. scikit-learn: for test/train splitting of dataset.
7. tqdm: for visualizing progress when colorizing images.

# Resources:
[1] S. Panchal, Colorizing B/W Images With GANs in TensorFlow, heartbeat, November 27, 2020. Accessed on: May 8, 2021. [Online]. Available: https://heartbeat.fritz.ai/colorizing-b-w- images-with-gans-in-tensorflow-f444f737db6c

[2] M. Shariatnia, Colorizing black & white images with U-Net and conditional GAN – A Tutorial, towards data science, November 18, 2020. Accessed on: May 9, 2021. [Online]. Available: https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and- conditional-gan-a-tutorial-81b2df111cd8

[3] Q. Fu, W. Hsu, and M. Yang, Colorization Using CovNet and GAN, Stanford University, n.d

[4] S. Anwar, M. Tahir, C. Li, A. Mian, F. Shabaz Khan, and A. Wahab Muzaffar, Image Colorization: A Survey and Dataset, arXiv:2008.10774v2 [cs.CV], November 3 2020