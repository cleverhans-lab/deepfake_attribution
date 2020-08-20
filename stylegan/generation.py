# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf
from collections import OrderedDict

SAVE_DIR = '../data/stylegan/'
SEED_DIR = '../data/random_seeds/generation'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Convert images to PIL-compatible format.
def process_images(images):
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    return images


# Initialize TensorFlow.
tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    G, _D, Gs = pickle.load(f)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

# Print network details.
# Gs.print_layers()


# Pick latent vector.
rnd = np.random.RandomState(5)

#num_images = 1000
#num_repeats = 10
latents = rnd.randn(1000, Gs.input_shape[1])
#latents = np.repeat(latents, num_repeats, axis=0)



# Generate image.
#fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
#images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

# Generate target
for seed_index in range(3000):
    print(seed_index)
    target_latent = np.load(os.path.join(SEED_DIR, str(seed_index)+'.npy'))
    target_images = Gs.run(target_latent, None, truncation_psi=0.7, randomize_noise=False)
    np.save(os.path.join(SAVE_DIR, '{}-target.npy'.format(str(seed_index))), target_images[0])
    #processed_target_images = tflib.convert_images_to_uint8(target_images, nchw_to_nhwc=True)
    processed_target_images = process_images(target_images)
    PIL.Image.fromarray(processed_target_images[0], 'RGB').save(os.path.join(SAVE_DIR, '{}-target.png'.format(str(seed_index))))


## Prepare Seed Reconstruction
#seed_latent = tf.Variable(latents[[SEED_INDEX]], dtype=tf.float32)
#lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
#noise_opt = tflib.Optimizer(name='Noise', learning_rate=lrate_in, beta1=0.0, beta2=0.99, epsilon=1e-8)
#target = tf.expand_dims(tf.convert_to_tensor(target_images[0]), 0)
#
#with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#    fake_images_out = G.get_output_for(seed_latent, None, is_training=False)
#    noise_loss = tf.keras.losses.MSE(target, fake_images_out)
#
#noise_opt.register_gradients(tf.reduce_mean(noise_loss), OrderedDict([('latents', seed_latent)]))
#noise_update_op = noise_opt.apply_updates()
#
#init_op = tf.variables_initializer([seed_latent])
#tf.get_default_session().run([init_op])
#
#image_losses = np.array([])
#latent_losses = np.array([])
#
#for i in range(STEPS):
#    _, reconstructed_latent, image_loss, out = tflib.run([noise_update_op, seed_latent, noise_loss, fake_images_out], {lrate_in: LR})
#    #print(np.mean(loss))
#
#    if i % LOG_INTERVAL == 0 or i == (STEPS-1):
#        image_loss = np.mean(image_loss)
#        latent_loss = ((target_latent - reconstructed_latent)**2).mean()
#        if i == 0:
#            image_losses = np.array([[i, image_loss]])
#            latent_losses = np.array([[i, latent_loss]]) 
#        else:
#            image_losses = np.concatenate((image_losses, [[i, image_loss]]), axis = 0)
#            latent_losses = np.concatenate((latent_losses, [[i, latent_loss]]), axis = 0)
#
#        out = process_images(out)
#        for idx in range(out.shape[0]):
#            PIL.Image.fromarray(out[idx], 'RGB').save(SAVE_DIR + '/reconstructed-%d.png' % (i))
#            print('Saved ' + SAVE_DIR + '/reconstructed-%d.png' % (i))
#
#np.save(SAVE_DIR+'/image_losses,npy', image_losses)
#np.save(SAVE_DIR+'/latent_losses.npy', latent_losses)
#
#
## Save images.
##os.makedirs(config.result_dir, exist_ok=True)
##for i in range(num_repeats):
##    png_filename = os.path.join(config.result_dir, 'example-0-no-random-{}.png'.format(str(i)))
##    PIL.Image.fromarray(images[i], 'RGB').save(png_filename)
#
#import matplotlib.pyplot as plt
#plt.subplot(1, 2, 1)
#plt.title('Image MSE Loss')
#plt.xlabel('Steps')
#plt.ylabel('Loss')
#plt.ylim([0, 1])
#plt.plot(image_losses[:, 0], image_losses[:, 1])
#
#plt.subplot(1, 2, 2)
#plt.title('Noise MSE Loss')
#plt.xlabel('Steps')
#plt.ylabel('Loss')
#plt.ylim([1, 3])
#plt.plot(latent_losses[:, 0], latent_losses[:, 1])
#plt.savefig(SAVE_DIR + '/losses.png')
#
#print('Saved ' + SAVE_DIR + '/losses.png')
#
