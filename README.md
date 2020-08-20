# Not My DeepFake

This is the demo code for **Not My Deepfake: Towards Plausible Deniability for Machine-Generated Media**.
This codebase is built upon official implementations of [ProgressiveGan](https://github.com/tkarras/progressive_growing_of_gans), [Stylegan](https://github.com/NVlabs/stylegan), and [Stylegan2](https://github.com/NVlabs/stylegan2).

## Requrirements
System requirements for all three GANs need to be satisfied. 
- Linux is recommended. We did not test on Windows systems. 
- 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
- TensorFlow 1.14 or 1.15 with GPU support. The code does not support TensorFlow 2.0.
- [Tensorflow Models](https://github.com/tensorflow/models) with Slim support. Set an environment variable to slim. For example: ```export PYTHONPATH="~/models/research/slim:~/models/research:$PYTHONPATH"```
- One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5.

Note: 
1. CUDA 10.0 toolkit and cuDNN 7.5 are required for [Stylegan2](https://github.com/NVlabs/stylegan2#requirements), but not for ProgressiveGan and Stylegan. 
2. Replace line 127 in `stylegan2/dnnlib/tflib/custom_ops.py` with `compile_opts += ' --compiler-options \'-fPIC\''` if you encounter compile errors for custom ops.
## Demo
To run the demo code, please follow steps below:
1. Download pretrained weights for three GANs to their desinated folders: [ProgressiveGan](https://drive.google.com/open?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4), [Stylegan](https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ), and [Stylegan2](http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl).
2. Download [Inception V3 Checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) to checkpoints folder. 
3. Run ```cd code; bash demo.sh```


