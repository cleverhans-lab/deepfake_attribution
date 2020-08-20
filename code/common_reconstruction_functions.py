import cv2
import os
import pickle
import numpy as np
import PIL.Image
import tensorflow as tf
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
from shutil import copyfile
import sys
import json
import argparse
from matplotlib.image import imread
from nets import nets_factory
import warnings
rnd = np.random.RandomState(2)

def generate_random_seed():
    return rnd.randn(1, 512)

def process_images(images):
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                     255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return images

def de_process_images(images):
    images = images / 255.0 * 2 - 1  # [0, 255] => [-1, 1]
    images = images.transpose(0, 3, 1, 2)  # NHWC => NCHW
    return images

def get_existing_seeds(path, output_format):
    all_files = [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    existing_seeds = []
    for file in all_files:
        s_index = file.split('_')[1]
        if int(s_index) not in existing_seeds:
            if output_format == 'original':
                if 'r_{}_losses.png'.format(s_index) in all_files and 'r_{}_log_losses.npz'.format(s_index) in all_files and 'r_{}_log_reconstruction_image.jpg'.format(s_index) in all_files:
                    existing_seeds.append(int(s_index))
                else:
                    os.remove(os.path.join(path, file))
            elif output_format == 'simp':
                existing_seeds.append(int(s_index))
            else:
                raise ValueError()
    return existing_seeds

# Function to reconstruct a single image
def reconstruct_single_image(target, target_np, lrate_in, seed_latent, seed_latent_np, args, sess, noise_update_op, noise_loss, fake_images_out, target_latent_np=None):
    target.load(target_np)
    seed_latent.load(seed_latent_np)
    log_reconstruction_image = None
    latent_losses = None
    for i in range(args.steps):
        _, reconstructed_latent, image_loss, out = sess.run(
            [noise_update_op, seed_latent, noise_loss, fake_images_out], {lrate_in: args.lr})
        # _, reconstructed_latent, image_loss, out, current_gs, current_lr = sess.run(
        #     [noise_update_op, seed_latent, noise_loss, fake_images_out, increment_global_step_op, learning_rate], )
        image_loss = np.mean(image_loss)
        if target_latent_np is not None:
            latent_loss = (
                (target_latent_np - reconstructed_latent)**2).mean()
        if i == 0:
            image_losses = np.array([[i, image_loss]])
            if target_latent_np is not None:
                latent_losses = np.array([[i, latent_loss]])
        else:
            image_losses = np.concatenate(
                (image_losses, [[i, image_loss]]), axis=0)
            if target_latent_np is not None:
                latent_losses = np.concatenate((latent_losses, [[i, latent_loss]]), axis=0)

        if i % args.image_log_interval == 0 or i == (args.steps-1):
            out = process_images(out)
            for idx in range(out.shape[0]):
                if log_reconstruction_image is None:
                    log_reconstruction_image = out[idx]
                else:
                    log_reconstruction_image = np.concatenate(
                        (log_reconstruction_image, out[idx]), axis=1)
    return image_losses, latent_losses, reconstructed_latent, log_reconstruction_image

def plot_and_save(image_losses, latent_losses, reconstructed_latent, log_reconstruction_image, SEED_INDEX, increament_suffix, TARGET_PATH, experiment_dir, mode='original'):
    # Original
    # mode = 'simp' # 'original'
    if mode == 'original':
        log_reconstruction_image = np.concatenate(
            (log_reconstruction_image, (255.0*imread(TARGET_PATH)).astype(np.uint8)), axis=1)
        # PIL.Image.fromarray(log_reconstruction_image, 'RGB').save(os.path.join(
        #     experiment_dir, 'r_{}_log_reconstruction_image{}.png'.format(str(SEED_INDEX), increament_suffix)))
        cv2.imwrite(os.path.join(experiment_dir, 'r_{}_log_reconstruction_image{}.jpg'.format(str(
            SEED_INDEX), increament_suffix)), cv2.cvtColor(log_reconstruction_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        print('Saved ' + os.path.join(experiment_dir,
                                    'r_{}_log_reconstruction_image{}.png'.format(str(SEED_INDEX), increament_suffix)))
        # print(image_losses.shape)
        np.savez(os.path.join(experiment_dir, 'r_{}_log_losses{}.npz'.format(
            str(SEED_INDEX), increament_suffix)), image_losses=image_losses, latent_losses=latent_losses, reconstructed_latent=reconstructed_latent)

        plt.subplot(1, 2, 1)
        plt.title('Image MSE Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.ylim([0, 0.7])
        plt.plot(image_losses[:, 0], image_losses[:, 1])

        if latent_losses is not None:
            plt.subplot(1, 2, 2)
            plt.title('Noise MSE Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            # plt.ylim([1, 3])
            plt.plot(latent_losses[:, 0], latent_losses[:, 1])
        plt.savefig(os.path.join(experiment_dir,
                                'r_{}_losses{}.png'.format(str(SEED_INDEX), increament_suffix)))

        print('Saved ' + os.path.join(experiment_dir,
                                    'r_{}_losses{}.png'.format(str(SEED_INDEX), increament_suffix)))
        plt.close()
        return experiment_dir
    elif mode == 'simp':
        last_image = log_reconstruction_image[:,-1024:,:]
        # cv2.imwrite(os.path.join(experiment_dir, 'r_{}_log_reconstruction_image{}.jpg'.format(str(
        #     SEED_INDEX), increament_suffix)), cv2.cvtColor(last_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        save_image_path = os.path.join(experiment_dir, 'r_{}_log_reconstruction_image{}_{}.png'.format(str(SEED_INDEX), increament_suffix, str(image_losses[-1][1])))
        PIL.Image.fromarray(last_image, 'RGB').save(save_image_path)
        return save_image_path

# Argument parser
def get_parser(name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--name', type=str,
                        help='experiment name')
    parser.add_argument('--num-repeats', type=int, default=3, metavar='N',
                        help='number of random seed for each image')
    parser.add_argument('--steps', type=int, default=500, metavar='N',
                        help='number of steps')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--loss-log-interval', type=int, default=1, metavar='N',
                        help='number of steps to log loss')
    parser.add_argument('--image-log-interval', type=int, default=300, metavar='N',
                        help='number of steps to log image')
    parser.add_argument('--target-model-name', type=str, default='',
                        help='target model name')
    parser.add_argument('--truncation-psi', type=float, default=0.7,
                        help='truncaton rate')
    parser.add_argument('--min-index', type=int, default=-1, metavar='N',
                        help='min index for target images')
    parser.add_argument('--max-index', type=int, default=-1, metavar='N',
                        help='max index for target images')
    parser.add_argument('--lin-int-num-segment', type=int, default=20,
                        help='Number of segments in liner interpolation. ')
    parser.add_argument('--checkpoint-path', type=str,
                        help='Checkpoint directories that stores TF Models checkpoints')
    parser.add_argument('--feature-extractor', type=str,
                        choices=['inception_v3', 'D', 'None'], help='Type of feature extractor can be used')
    parser.add_argument('--feature-extractor-layer', type=str,
                        default='Mixed_7a', help='Feature extractor layer')
    parser.add_argument('--adv', action='store_true',
                        default=False, help='Using adv example')
    parser.add_argument('--adv-prefix', type=str,
                        default='fgsm_01', help='Adv example prefix')
    parser.add_argument('--prefix', type=str, default='original',
                        help='Target folder prefix')
    parser.add_argument('--run-mode', type=str,
                        choices=['original_run', 'folder_run', 'single_run', 'demo_run'], default='original_run', help='Type of running mode.')
    parser.add_argument('--target-folder', type=str, default='', help='Target folder in folder-run mode.')
    parser.add_argument('--target-image-path', type=str, default='', help='Target image in single-run mode.')
    parser.add_argument('--output-format', type=str,choices=['original', 'simp'],  default='original', help='Output format')
    args = parser.parse_args()
    return args

def get_nosie_output(args, Gs, seed_latent, seed_label, target, feture_extractor):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images_out = Gs.get_output_for(
            seed_latent, seed_label, is_training=False, randomize_noise=False)
        if args.feature_extractor != 'None':
            # Use external feature extractor
            if args.feature_extractor != 'D':
                image_to_extract = tf.concat([fake_images_out, target], 0)
                image_to_extract_nhwc = tf.transpose(
                    image_to_extract, [0, 2, 3, 1])
                image_to_extract_nhwc_resized = tf.image.resize(
                    image_to_extract_nhwc, [299, 299])
                image_to_extract_resized = tf.transpose(
                    image_to_extract_nhwc_resized, [0, 3, 1, 2])
                logits, endpoints = feture_extractor(
                    image_to_extract_nhwc_resized)
                fake_image_features, target_image_features = tf.unstack(
                    endpoints[args.feature_extractor_layer])
                noise_loss = tf.keras.losses.MSE(
                    fake_image_features, target_image_features)
            # Use internal D as feature extractor
            else:
                raise ValueError('This part is not tailored for each discriminator. DO NOT USE IT.')
                image_to_extract = tf.concat([fake_images_out, target], 0)
                image_to_extract_nhwc = tf.transpose(
                    image_to_extract, [0, 2, 3, 1])
                D_out = D.get_output_for(
                    image_to_extract, tf.zeros([2, 0]), is_training=False)
                tensor_candidates = [n.name for n in tf.get_default_graph().as_graph_def().node if 'D_1' in n.name and '{}x{}'.format(
                    str(args.feature_extractor_layer), str(args.feature_extractor_layer)) in n.name and '/IdentityN' in n.name and '/Conv0' in n.name]
                if len(tensor_candidates) != 1:
                    print(tensor_candidates)
                    raise ValueError()
                D_feature = tf.get_default_graph().get_tensor_by_name(
                    tensor_candidates[0]+":0")
                fake_image_features, target_image_features = tf.unstack(
                    D_feature)
                noise_loss = tf.keras.losses.MSE(
                    fake_image_features, target_image_features)
        else:
            noise_loss = tf.keras.losses.MSE(target, fake_images_out)
    return noise_loss, fake_images_out

def get_feature_extractor(args):
    if args.feature_extractor != 'None' and args.feature_extractor != 'D':
        feature_extractor = nets_factory.get_network_fn(
            args.feature_extractor, num_classes=1001, is_training=False)
        return feature_extractor
    else:
        return None

def safe_makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            return True
        except:
            print('oops')
            return False
    else:
        return False

def original_run(args, TARGET_DIR, SAVE_DIR, SEED_DIR, TARGET_SEED_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out, output_format='original'):

    # Run with given target indices
    if args.min_index != -1 and args.max_index != -1:
        target_indices = range(args.min_index, args.max_index)

    # Run with all targets images in TARGET_DIR
    else:
        target_image_paths = os.listdir(TARGET_DIR)
        target_indices = [int(x.split('-')[0])
                          for x in target_image_paths if x.endswith('.npy')]
        target_indices.sort()

    for TARGET_INDEX in target_indices:

        experiment_dir = os.path.join(
            SAVE_DIR, 't-{}'.format(str(TARGET_INDEX)))
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        target_np = de_process_images(np.expand_dims(255.0 * imread(os.path.join(
            TARGET_DIR, str(TARGET_INDEX)+'-target.png')), axis=0))

        target_latent_np = np.load(os.path.join(
            TARGET_SEED_DIR, str(TARGET_INDEX)+'.npy'))

        existing_seeds = get_existing_seeds(experiment_dir, args.output_format)

        for num_repeat in range(args.num_repeats - len(existing_seeds)):
            SEED_INDEX = random.randint(0, 999)
            while SEED_INDEX in existing_seeds:
                SEED_INDEX = random.randint(0, 999)
            existing_seeds.append(SEED_INDEX)

            print('target_index: ', TARGET_INDEX,
                  'seed_index:', SEED_INDEX)
            seed_latent_np = np.load(os.path.join(
                SEED_DIR, str(SEED_INDEX)+'.npy'))

            for increment_index in range(args.lin_int_num_segment+1):
                if increment_index == 0:
                    increament_suffix = ''
                else:
                    increament_suffix = '_{}'.format(increment_index)

                if args.lin_int_num_segment != 0:
                    cur_seed_latent_np = seed_latent_np + increment_index / \
                        float(args.lin_int_num_segment) * \
                        (target_latent_np - seed_latent_np)
                else:
                    cur_seed_latent_np = seed_latent_np
                print('Init seed difference: ',
                      ((target_latent_np - cur_seed_latent_np)**2).mean())

                image_losses, latent_losses, reconstructed_latent, log_reconstruction_image = reconstruct_single_image(
                    target, target_np, lrate_in, seed_latent, cur_seed_latent_np, args, sess, noise_update_op, noise_loss, fake_images_out, target_latent_np)

                print('Ori var:', np.var(target_latent_np),
                      'Gen var:', np.var(reconstructed_latent))
                print('Ori range:', np.min(seed_latent_np), np.max(seed_latent_np), 'Rec range:', np.min(
                    reconstructed_latent), np.max(reconstructed_latent))

                plot_and_save(image_losses, latent_losses, reconstructed_latent, log_reconstruction_image, SEED_INDEX, increament_suffix, os.path.join(TARGET_DIR, str(TARGET_INDEX)+'-target.png'), experiment_dir)

def folder_run(args, SAVE_DIR, SEED_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out):

    target_image_paths = [f for f in os.listdir(args.target_folder) if f.endswith('.png') ]

    for target_image_path in target_image_paths:

        experiment_dir = os.path.join(
            SAVE_DIR, target_image_path.split('.')[0])
        safe_makedir(experiment_dir)
        target_np = de_process_images(np.expand_dims(255.0 * imread(os.path.join(
            args.target_folder, target_image_path)), axis=0))

        existing_seeds = get_existing_seeds(experiment_dir)

        for num_repeat in range(args.num_repeats - len(existing_seeds)):
            SEED_INDEX = random.randint(0, 999)
            while SEED_INDEX in existing_seeds:
                SEED_INDEX = random.randint(0, 999)
            existing_seeds.append(SEED_INDEX)

            seed_latent_np = np.load(os.path.join(
                SEED_DIR, str(SEED_INDEX)+'.npy'))

            image_losses, _, reconstructed_latent, log_reconstruction_image = reconstruct_single_image(
                target, target_np, lrate_in, seed_latent, seed_latent_np, args, sess, noise_update_op, noise_loss, fake_images_out, None)


            plot_and_save(image_losses, None, reconstructed_latent, log_reconstruction_image, SEED_INDEX, '', os.path.join(args.target_folder, target_image_path), experiment_dir)

def single_run(args, SAVE_DIR, SEED_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out):

    target_image_path = args.target_image_path
    if '/' in target_image_path:
        target_image_name = target_image_path.split('/')[-1].split('.')[0]
    else:
        target_image_name = target_image_path.split('.')[0]

    experiment_dir = os.path.join(
        SAVE_DIR, target_image_name)
    if not safe_makedir(experiment_dir):
        warnings.warn(experiment_dir+' already exist! Abort! Image name must be unique.')
    target_np = de_process_images(np.expand_dims(255.0 * imread(os.path.join(target_image_path)), axis=0))

    existing_seeds = get_existing_seeds(experiment_dir)

    for num_repeat in range(args.num_repeats - len(existing_seeds)):
        SEED_INDEX = random.randint(0, 999)
        while SEED_INDEX in existing_seeds:
            SEED_INDEX = random.randint(0, 999)
        existing_seeds.append(SEED_INDEX)

        seed_latent_np = np.load(os.path.join(
            SEED_DIR, str(SEED_INDEX)+'.npy'))

        image_losses, _, reconstructed_latent, log_reconstruction_image = reconstruct_single_image(
            target, target_np, lrate_in, seed_latent, seed_latent_np, args, sess, noise_update_op, noise_loss, fake_images_out, None)


        plot_and_save(image_losses, None, reconstructed_latent, log_reconstruction_image, SEED_INDEX, '', target_image_path, experiment_dir)

# Run on a single image for demo purpose
def demo_run(args, SAVE_DIR, target, lrate_in, seed_latent, sess, noise_update_op, noise_loss, fake_images_out):

    target_image_path = args.target_image_path
    if '/' in target_image_path:
        target_image_name = target_image_path.split('/')[-1].split('.')[0]
    else:
        target_image_name = target_image_path.split('.')[0]

    experiment_dir = os.path.join(
        SAVE_DIR, target_image_name)
    if not safe_makedir(experiment_dir):
        warnings.warn(experiment_dir+' already exist! Abort! Image name must be unique.')
    target_np = de_process_images(np.expand_dims(255.0 * imread(os.path.join(target_image_path)), axis=0))
    results = []
    for num_repeat in range(args.num_repeats):
        # SEED_INDEX = random.randint(0, 999)
        # while SEED_INDEX in existing_seeds:
        #     SEED_INDEX = random.randint(0, 999)
        # existing_seeds.append(SEED_INDEX)

        # seed_latent_np = np.load(os.path.join(
        #     SEED_DIR, str(SEED_INDEX)+'.npy'))
        print('Repeat: ', num_repeat)
        seed_latent_np = generate_random_seed()

        image_losses, _, reconstructed_latent, log_reconstruction_image = reconstruct_single_image(
            target, target_np, lrate_in, seed_latent, seed_latent_np, args, sess, noise_update_op, noise_loss, fake_images_out, None)


        reconstruction_path = plot_and_save(image_losses, None, reconstructed_latent, log_reconstruction_image, str(num_repeat), '', target_image_path, experiment_dir)

        results.append((image_losses[-1][1], reconstruction_path))

    results.sort()
    return results[0]
