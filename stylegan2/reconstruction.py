# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import cv2
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
from shutil import copyfile
import pretrained_networks
from tensorflow.python.client import timeline
import sys
import json
import argparse
from matplotlib.image import imread
from nets import nets_factory
slim = tf.contrib.slim
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.append('../code')
from common_reconstruction_functions import *

SEED_DIR = '../data/random_seeds/reconstruction'

def load_network():
    # network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    # print('Loading networks from "%s"...' % network_pkl)
    # _G, D, Gs = pretrained_networks.load_networks(network_pkl)
    # return Gs
    with open('stylegan2-ffhq-config-f.pkl', 'rb') as file:
        G, D, Gs = pickle.load(file)
    return Gs

if __name__ == "__main__":
    args = get_parser('Stylegan 2 reconstruction')

    # Initialize TensorFlow.
    tflib.init_tf()
    _global_step = tf.train.get_or_create_global_step()


    # Define Variables 
    seed_latent = tf.Variable(np.zeros((1, 512)), dtype=tf.float32)
    target = tf.Variable(np.zeros((1, 3, 1024, 1024)), dtype=np.float32)


    # Define network 
    Gs = load_network()
    fea_ext_name_to_var_name = {'inception_v3': 'InceptionV3'}
    feature_extractor = get_feature_extractor(args)
    noise_loss, fake_images_out = get_nosie_output(args, Gs, seed_latent, None, target, feature_extractor)
    

    # Apply optimizers
    lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
    noise_opt = tflib.Optimizer(
        name='Noise', learning_rate=lrate_in, beta1=0.0, beta2=0.99, epsilon=1e-8)
    noise_opt.register_gradients(tf.reduce_mean(
        noise_loss), OrderedDict([('latents', seed_latent)]))
    noise_update_op = noise_opt.apply_updates()

    # Get TF Session
    sess = tf.get_default_session()
    
    # Initialize variables
    uninstizalized_vars = [seed_latent, target]
    init_op = tf.variables_initializer(uninstizalized_vars)
    sess.run([init_op])
    if args.feature_extractor is not None and args.feature_extractor != 'D':
        extractor_variables = slim.get_variables_to_restore()
        extractor_variables = [
            x for x in extractor_variables if fea_ext_name_to_var_name[args.feature_extractor] in x.name]
        restore_fn = slim.assign_from_checkpoint_fn(
            args.checkpoint_path, extractor_variables)
        restore_fn(sess)


    # Run reconstruction
    if args.run_mode == 'original_run':
        SAVE_DIR = '../data/experiments/{}/reconstruction/model_stylegan2/image_'.format(args.name) + args.target_model_name
        safe_makedir(SAVE_DIR)
        TARGET_DIR = '../data/target_images/{}/{}'.format(args.prefix, args.target_model_name)
        TARGET_SEED_DIR = '../data/random_seeds/generation'

        original_run(args, TARGET_DIR, SAVE_DIR, SEED_DIR, TARGET_SEED_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out)
    elif args.run_mode == 'folder_run':
        SAVE_DIR = '../data/experiments/{}/reconstruction/model_stylegan2/'.format(args.name)
        safe_makedir(SAVE_DIR)
        folder_run(args, SAVE_DIR, SEED_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out)
    elif args.run_mode == 'single_run':
        SAVE_DIR = '../data/experiments/{}/reconstruction/model_stylegan2/'.format(args.name)
        safe_makedir(SAVE_DIR)
        single_run(args, SAVE_DIR, SEED_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out)
    elif args.run_mode == 'demo_run':
        SAVE_DIR = '../experiments/{}/reconstruction/model_stylegan2/'.format(args.name)
        safe_makedir(SAVE_DIR)
        demo_run(args, SAVE_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out)
    else:
        raise ValueError('Not implemented yet!')