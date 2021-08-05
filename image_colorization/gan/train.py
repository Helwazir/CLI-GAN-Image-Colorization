import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanSquaredError
from generator import define_generator
from discriminator import define_discriminator
from image_colorization import dataset 


def discriminator_loss(real_output, fake_output):
    real_loss = BinaryCrossentropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1), real_output)
    fake_loss = BinaryCrossentropy(tf.zeros_like(fake_output) - tf.random.uniform(shape=fake_output.shape, maxval=0.1), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, real_y):
    real_y = tf.cast(real_y, 'float32')
    return MeanSquaredError(fake_output, real_y)

@tf.function
def train_step(input_x, real_y):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate a colorized image
        generated_images = generator(input_x, training=True)
        # Probability that the image is real
        real_output = discriminator(real_y, training=True)
        # Probability that the image is the colorized image
        generated_output = discriminator(generated_images, training=True)
        # Calculate and log losses
        gen_loss = generator_loss(generated_images, real_y)
        disc_loss = discriminator_loss(real_output, generated_output)
        
    # Calculate gradients
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Apply gradients with optimizers
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

def train(dataset_dir, epochs, save=False, name=None):
    # Define optimizers
    generator_optimizer = Adam(0.001)
    discriminator_optimizer = Adam(0.001)
    # Define generator and discriminator
    generator = define_generator_model()
    discriminator = rev4_discrimiator_model()
    # Prepare dataset for training
    dataset = dataset.prepare_dataset(dataset_dir)
    # Manually step through epochs and batches
    for e in range(epochs + 1):
        for (x, y) in dataset:
            train_step(x, y)
    if save and name:
        if name:
            generator.save(name)
        else:
            print('Enter name to save generator')
