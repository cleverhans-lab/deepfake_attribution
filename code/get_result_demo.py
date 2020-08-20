import numpy as np
import os

def get_min_loss(path):
    for image in os.listdir(path):
        results = []
        for f in os.listdir(os.path.join(path, image)):
            if f.endswith('.npz'):
                image_losses = np.load(os.path.join(path, image, f))['image_losses']
                end_value = image_losses[-1][1]
                results.append((end_value, image, path, 'r_{}_log_reconstruction_image.jpg'.format(f.split('_')[1])))
        results.sort()
        if image not in image_results:
            image_results[image] = [results[0]]
        else:
            image_results[image].append(results[0])

    # return results[0]

demo_result_path = '../experiments/test/reconstruction/'
image_results = {}

if __name__ == "__main__":
    results= []
    for folder in os.listdir(demo_result_path):
        get_min_loss(os.path.join(demo_result_path, folder))

    for image in image_results.keys():
        image_result = image_results[image]
        # print(image_result)
        image_result.sort()
        print()
        print('Attribution:',  image_result[0][1], '->', image_result[0][2].split('/')[-1])
        print('       Loss: {:.3f}'.format(image_result[0][0]))
        print('    Results:', (os.path.join(image_result[0][2], image_result[0][1], image_result[0][3])))
