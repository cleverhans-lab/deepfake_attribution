# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os

import pretrained_networks

# Convert images to PIL-compatible format.
def process_images(images):
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    return images


SAVE_DIR = '../data/target_images/stylegan2_07'
SEED_DIR = '../data/random_seeds/generation'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
seeds = range(100)
truncation_psi = 0.7

print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

Gs_kwargs = dnnlib.EasyDict()
# Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
if truncation_psi is not None:
    Gs_kwargs.truncation_psi = truncation_psi

for seed_idx, seed in enumerate(seeds):
    print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    rnd = np.random.RandomState(seed)
    # z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    # print(z.shape)
    # exit()
    target_latent = np.load(os.path.join(SEED_DIR, str(seed_idx)+'.npy'))
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    target_images = Gs.run(target_latent, None, **Gs_kwargs) # [minibatch, height, width, channel]
    np.save(os.path.join(SAVE_DIR, '{}-target.npy'.format(str(seed_idx))), target_images[0])
    processed_target_images = process_images(target_images)
    PIL.Image.fromarray(processed_target_images[0], 'RGB').save(os.path.join(SAVE_DIR, '{}-target.png'.format(str(seed_idx))))
