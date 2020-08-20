import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import config
import tfutil
import dataset
import misc
from collections import OrderedDict
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
sys.path.append('../code')
from common_reconstruction_functions import *

#TARGET_INDEX = 415
#SEED_INDEX = 86
#EXPERIMENT_NAME = 'test-{}-{}'.format(str(TARGET_INDEX), str(SEED_INDEX))
#STEPS = 3000
#LR = 3e-3
#LOG_INTERVAL = 100
SEED_DIR = '../data/random_seeds/generation'
SAVE_DIR = '../data/progressive/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Convert images to PIL-compatible format.
def process_images(images):
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    return images




# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

## Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
##latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

# # Generate Target
# for seed_index in range(3000):
#     print(seed_index)
#     target_latent = np.load(os.path.join(SEED_DIR, str(seed_index)+'.npy'))
#     target_label = np.zeros([target_latent.shape[0]] + Gs.input_shapes[1][1:], dtype=np.float32)
#     target_images = Gs.run(target_latent, target_label)
#     np.save(os.path.join(SAVE_DIR, '{}-target.npy'.format(str(seed_index))), target_images[0])
#     processed_target_images = process_images(target_images)
#     PIL.Image.fromarray(processed_target_images[0], 'RGB').save(os.path.join(SAVE_DIR, '{}-target.png'.format(str(seed_index))))

image_dirs = ['../data/target_images/original/stylegan_100/', '../data/target_images/original/stylegan2_07_100/', '../data/target_images/original/progressive_100/']

for image_dir in image_dirs:
    scores = []
    for i in range(100):
        target_image_path = os.path.join(image_dir, '{}-target.png'.format(str(i)))
        # target_image_path = os.path.join(REAL_DIR, '{}.png'.format(real_index[i]))
        target_np = de_process_images(np.expand_dims(255.0 * imread(target_image_path), axis=0))
        # print(np.min(target_np), np.max(target_np))
        output_score, _ = D.run(target_np)
        # print(target_image_path, output_score)
        scores.append(output_score)
    scores = np.array(scores)
    print(image_dir, 'min: ', np.min(scores), 'max: ', np.max(scores),'avg: ', np.average(scores),'median: ', np.median(scores))

## Prepare Seed Reconstruction
#seed_latent = tf.Variable(latents[[SEED_INDEX]], dtype=tf.float32)
#seed_label = np.zeros([seed_latent.shape[0]] + Gs.input_shapes[1][1:], dtype=np.float32)
#lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
#noise_opt = tfutil.Optimizer(name='Noise', learning_rate=lrate_in, **config.G_opt)
#target = tf.expand_dims(tf.convert_to_tensor(target_images[0]), 0)
#
#
#
#with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#    fake_images_out = G.get_output_for(seed_latent, seed_label, is_training=False)
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
#for i in range(STEPS):
#    _, reconstructed_latent, image_loss, out = tfutil.run([noise_update_op, seed_latent, noise_loss, fake_images_out], {lrate_in: LR})
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


#print(out.shape)
##images = Gs.run(ori_latents, ori_labels)
#images = G.run(new_latents, labels.tolist())
#print('Output max and min:', np.max(images), np.min(images))
#print(images.shape)
#temp = np.load('example_script_outputs/target0.npy')
#print('Target max and min:', np.max(temp), np.min(temp))
#
## # Save images as npy.
## for idx in range(images.shape[0]):
##    np.save('example_script_outputs/target%d.npy' % (idx), images[idx])
#
## Convert images to PIL-compatible format.
#images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
#images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
#
## Save images as PNG.
#for idx in range(images.shape[0]):
#   PIL.Image.fromarray(images[idx], 'RGB').save('example_script_outputs/reconstructed%d.png' % (idx))
