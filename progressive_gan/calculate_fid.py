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


SEED_DIR = '../data/random_seeds/generation'
SAVE_DIR = '../data/progressive/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Convert images to PIL-compatible format.
def process_images(images):
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    return images

def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False, shrink=1):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        ksize = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return tf.saturate_cast(images, tf.uint8)



# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

## Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:])

target_latent = np.load(os.path.join(SEED_DIR, str(0)+'.npy'))
target_label = np.zeros([target_latent.shape[0]] + Gs.input_shapes[1][1:], dtype=np.float32)

with open('inception_v3_features.pkl', 'rb') as file:
    inception = pickle.load(file, encoding='latin1')
# inception = pickle.load('inception_v3_features.pkl')
# inception = load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')

minibatch_per_gpu=8
num_gpus = 1
minibatch_size = num_gpus * minibatch_per_gpu

# Construct TensorFlow graph.
result_expr = []
for gpu_idx in range(num_gpus):
    with tf.device('/gpu:%d' % gpu_idx):
        Gs_clone = Gs.clone()
        inception_clone = inception.clone()
        latents = tf.random_normal([minibatch_per_gpu] + Gs_clone.input_shape[1:])
        target_label = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:], dtype=np.float32)
        images = Gs_clone.get_output_for(latents, target_label)
        images = convert_images_to_uint8(images)
        result_expr.append(inception_clone.get_output_for(images))

activations = np.empty([50000, inception.output_shape[1]], dtype=np.float32)

# Calculate statistics for fakes.
for begin in range(0, 50000, minibatch_size):
# for begin in range(0, 640, minibatch_size):
    end = min(begin + minibatch_size, 50000)
    # print(begin, end)
    # exit()
    activations[begin:end] = np.concatenate(tf.get_default_session().run(result_expr), axis=0)[:end-begin]
    if end % 5000 == 0 or end in [1000, 2000, 3000, 4000]:

        mu_fake = np.mean(activations[:end], axis=0)
        sigma_fake = np.cov(activations[:end], rowvar=False)
        

        # m = np.square(mu_fake - mu_real).sum()
        # s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        # dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        print(begin, end)
        # print(np.real(dist))
        np.savez('fid_original_{}.npz'.format(str(end)), mu_fake=mu_fake, sigma_fake=sigma_fake)

# Calculate FID.
# m = np.square(mu_fake - mu_real).sum()
# s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
# dist = m + np.trace(sigma_fake + sigma_real - 2*s)
# print(np.real(dist))

#----------------------------------------------------------------------------



# # Generate Target
# for seed_index in range(3000):
#     print(seed_index)
#     target_latent = np.load(os.path.join(SEED_DIR, str(seed_index)+'.npy'))
#     target_label = np.zeros([target_latent.shape[0]] + Gs.input_shapes[1][1:], dtype=np.float32)
#     target_images = Gs.run(target_latent, target_label)
#     np.save(os.path.join(SAVE_DIR, '{}-target.npy'.format(str(seed_index))), target_images[0])
#     processed_target_images = process_images(target_images)
#     PIL.Image.fromarray(processed_target_images[0], 'RGB').save(os.path.join(SAVE_DIR, '{}-target.png'.format(str(seed_index))))