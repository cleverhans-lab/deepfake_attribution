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
from matplotlib.image import imread
import sys
sys.path.append('../code')
from common_reconstruction_functions import *

# SAVE_DIR = '../data/stylegan/'
# IMAGE_DIR = '../data/target_images/original/stylegan_100/'
# IMAGE_DIR = '../data/target_images/original/stylegan2_07_100/'
IMAGE_DIR = '../data/target_images/original/progressive_100/'
REAL_DIR = '../../hdd/datasets/ffhq-dataset/images1024x1024/00000/'
real_index = ['00002',  '00075' , '00113' , '00191' , '00219' , '00272' ,'00362']
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)


# Initialize TensorFlow.
tflib.init_tf()

# Load pre-trained network.
url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    G, _D, Gs = pickle.load(f)


# for i in range(10):
#     # target_image_path = os.path.join(IMAGE_DIR, '{}-target.png'.format(str(i)))
#     target_image_path = os.path.join(REAL_DIR, '{}.png'.format(real_index[i]))
#     target_np = de_process_images(np.expand_dims(255.0 * imread(target_image_path), axis=0))
#     # print(np.min(target_np), np.max(target_np))
#     output_score = _D.run(target_np, None, is_training=False)
#     print(target_image_path, output_score)


    # target_latent = np.load(os.path.join(SEED_DIR, str(seed_index)+'.npy'))
    # target_images = Gs.run(target_latent, None, truncation_psi=0.7, randomize_noise=False)
    # np.save(os.path.join(SAVE_DIR, '{}-target.npy'.format(str(seed_index))), target_images[0])
    # #processed_target_images = tflib.convert_images_to_uint8(target_images, nchw_to_nhwc=True)
    # processed_target_images = process_images(target_images)
    # PIL.Image.fromarray(processed_target_images[0], 'RGB').save(os.path.join(SAVE_DIR, '{}-target.png'.format(str(seed_index))))

image_dirs = ['../data/target_images/original/stylegan_100/', '../data/target_images/original/stylegan2_07_100/', '../data/target_images/original/progressive_100/']

for image_dir in image_dirs:
    scores = []
    for i in range(100):
        target_image_path = os.path.join(image_dir, '{}-target.png'.format(str(i)))
        # target_image_path = os.path.join(REAL_DIR, '{}.png'.format(real_index[i]))
        target_np = de_process_images(np.expand_dims(255.0 * imread(target_image_path), axis=0))
        # print(np.min(target_np), np.max(target_np))
        output_score = _D.run(target_np, None, is_training=False)
        # print(target_image_path, output_score)
        scores.append(output_score)
    scores = np.array(scores)
    print(image_dir, 'min: ', np.min(scores), 'max: ', np.max(scores),'avg: ', np.average(scores),'median: ', np.median(scores))