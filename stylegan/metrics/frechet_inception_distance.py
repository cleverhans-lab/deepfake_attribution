# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib
# import tensorflow_addons as tfa

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl') #'https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn') # inception_v3_features.pkl
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # print(cache_file)
        # exit()
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)            
            # np.savez('fid_real.npz', mu_real=mu_real, sigma_real=sigma_real)
            # exit()
        else:
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        # Augmentation 
        AUG = None #'gaussian_noise' #'zoom_in' #'flip' #'zoom_in'

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                images = Gs_clone.get_output_for(latents, None, is_validation=True, randomize_noise=True)
                images = tflib.convert_images_to_uint8(images)
                # result_expr.append(inception_clone.get_output_for(images))
                # tf.image.flip_left_right(image)

                # print(images)
                # exit()
                # nchw_to_nhwc
                print(images)
                images = tf.transpose(images, [0, 2, 3, 1])
                images = tf.saturate_cast(images, tf.float32)
                print(images)
                if AUG == 'flip':
                    augmented_images = tf.image.flip_left_right(images)
                elif AUG == 'zoom_in':
                    augmented_images = tf.image.central_crop(images, 0.9)
                    augmented_images = tf.image.resize(augmented_images, [1024, 1024])
                elif AUG =='random_rotate':
                    raise ValueError('Tensorflow addons is not available for this version.')
                elif AUG == 'random_crop':
                    pass
                elif AUG == 'gaussian_noise':
                    augmented_images = images + tf.random.normal(tf.shape(images), stddev=0.1)
                elif AUG =='random_crop':
                    pass
                    #augmented_images
                else:
                    augmented_images = images

                augmented_images = tf.saturate_cast(augmented_images, tf.uint8)
                # nhwc_to_nchw
                augmented_images = tf.transpose(augmented_images, [0, 3, 1, 2])

                result_expr.append(inception_clone.get_output_for(augmented_images))

        # Calculate statistics for fakes.
        for begin in range(0, self.num_images, minibatch_size):
        # for begin in range(0, 640, minibatch_size):
            end = min(begin + minibatch_size, self.num_images)
            # print(begin, end)
            # exit()
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
            if end % 5000 == 0 or end in [1000, 2000, 3000, 4000]:

                mu_fake = np.mean(activations[:end], axis=0)
                sigma_fake = np.cov(activations[:end], rowvar=False)
                

                m = np.square(mu_fake - mu_real).sum()
                s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
                dist = m + np.trace(sigma_fake + sigma_real - 2*s)
                print(begin, end)
                print(np.real(dist))
                np.savez('fid_original_{}_{}.npz'.format(str(end), AUG), mu_fake=mu_fake, sigma_fake=sigma_fake, fid_wrt_real=np.real(dist), activations=activations[:end])

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist))

#----------------------------------------------------------------------------
