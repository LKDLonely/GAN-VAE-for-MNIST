import tensorflow as tf
# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8), device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)




import glob
import imageio
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import IPython
from IPython import display




LR = 1e-6
generator_optimizer = tf.keras.optimizers.Adam(LR)
discriminator_optimizer = tf.keras.optimizers.Adam(LR)
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 5
epochsToSave = 1000000
noise_dim = 100
num_examples_to_generate = 16

tf.random.set_seed(42)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5



train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(10000).batch(1)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generator_model()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()



cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)




# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))





seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    # kl = tf.keras.losses.KLDivergence()
    # kl_div = tf.keras.backend.print_tensor(kl(real_output, fake_output))
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss
    # return gen_loss, disc_loss, kl_div


def train(dataset, epochs):
  # gen_losses, disc_losses, kl_divs = list(), list(), list()
  gen_losses, disc_losses = list(), list()
  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      # train_step(image_batch)
      # gen_loss, disc_loss, kl_div = train_step(image_batch)
      gen_loss, disc_loss = train_step(image_batch)
      gen_losses.append(gen_loss)
      disc_losses.append(disc_loss)
      # kl_divs.append(kl_div)


    if (epoch + 1) % epochsToSave == 0:
      # checkpoint.save(file_prefix = checkpoint_prefix)
      generator.save_weights("./GAN/generator")
      discriminator.save_weights("./GAN/discriminator")

    # generate_and_save_images(generator,
    #                          epoch,
    #                          seed)

    print(f'Time for epoch {epoch + 1} is {time.time()-start} sec. gen_loss: {gen_loss:.5f}, disc_loss: {disc_loss:.5f}')
    # print(f'Time for epoch {epoch + 1} is {time.time()-start} sec. gen_loss: {gen_loss:.5f}, disc_loss: {disc_loss:.5f}, kl_div: {kl_div:.5f}')
    # print(f'Time for epoch {epoch + 1} is {time.time()-start} ')


  # display.clear_output(wait=True)
  # generate_and_save_images(generator,
  #                          epochs,
  #                          seed)
  return gen_losses, disc_losses

def generate_and_save_images(model, epoch, test_input):

  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()


def generate_images(model, number, test_input):

  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_{:04d}.png'.format(number))



@tf.function
def test_step(images):
    predict = discriminator(images, training=True)
    return float(predict[0])

def test(dataset):
    correctNum = float(0)
    all = float(0)
    for image_batch in dataset:
      if(test_step(image_batch)>0):
        correctNum+=1
        all+=1
      else:
        all+=1
    accurancy = correctNum/all

    print("Number of Images Determined Real", correctNum)
    print("Number of all Images", all)
    return accurancy


try:
    generator.load_weights("./GAN/generator")
    discriminator.load_weights("./GAN/discriminator")
    print("Successfully Load Weights")
except:
    print("No Saved Weights of Models")

# exec part for train
# train(train_dataset, EPOCHS)
gen_losses, disc_losses = train(train_dataset, EPOCHS)
plt.figure(figsize=(10, 7))
plt.plot(gen_losses)
plt.plot(disc_losses)
plt.show()

# exec part for test
# for x in range(0,1):
#     accurancy = test(test_dataset)
#     print("accurancy:", accurancy)

# exec part for generate images
# for x in range(0,100):
#     tempseed = tf.random.normal([num_examples_to_generate, noise_dim])
#     display.clear_output(wait=True)
#     generate_images(generator,
#                      x,
#                      tempseed)