import numpy as np

from img_manipulations import *
import random
import warnings

warnings.filterwarnings('ignore')

def run_attacks(sample_list, attack_func, database_x, database_y, cnn_model, pca_model, dnn_model):
    '''
    Run attack and return accuracy as attack varies 
    '''
    cnn_acc = np.zeros(len(sample_list))
    pca_acc = np.zeros(len(sample_list))
    dnn_acc = np.zeros(len(sample_list))


    for i, val in np.ndenumerate(sample_list):
        cnn_acc[i] = cnn_model.evaluate( (attack_func(database_x, 0, val)), database_y)
        pca_acc[i] = pca_model.evaluate(attack_func(database_x, 0, val), database_y)
        dnn_acc[i] = dnn_model.evaluate(attack_func(database_x, 0, val), database_y)


    return cnn_acc, pca_acc, dnn_acc



def generate_spoofed_dataset(database_x, database_y):
    """
    Generates a list of same length and labels as x but with random filter functions
    applied to each element

    database_y should NOT be categorical, just the regular database_y
    
    """
    spoofed_dataset = np.array([])

    #distributions of filters (relative values)
    rotation_odd = 1
    gaussian_blur_odd = 1
    box_blur_odd = 1
    uniform_noise_odd = 1
    perlin_noise_odd = 1
    flip_image_odd = 1

    #quirks of filter that need them
    rotation_angle = 20
    gaussian_blur_sigma = 1
    box_blur_kernel = 2
    uniform_max_noise = 0.3
    perlin_max_noise = 0.3

    total = rotation_odd + gaussian_blur_odd + box_blur_odd + \
            uniform_noise_odd + perlin_noise_odd + flip_image_odd

    for i, image in np.ndenumerate(database_x):
        rand = random.random() * total

        if database_y[i] not in (6, 9): #avoid rotating the numbers 6 and 9 

            if rand <= rotation_odd/total : 
                np.append(spoofed_dataset, rotate_image(image, rotation_angle))

            elif rand <= (rotation_odd + gaussian_blur_odd)/total:
                np.append(spoofed_dataset, gaussian_blur(image, gaussian_blur_sigma))

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd)/total:
                np.append(spoofed_dataset, box_blur(image, box_blur_kernel) )

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd)/total:
                np.append(spoofed_dataset, uniform_noise(image, uniform_max_noise))

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd + perlin_noise_odd)/total:
                np.append(spoofed_dataset, perlin_noise(image, perlin_max_noise))

            else:
                np.append(spoofed_dataset, flip_image(image))


        else:
            if rand <= (rotation_odd + gaussian_blur_odd)/total:
                np.append(spoofed_dataset, gaussian_blur(image, gaussian_blur_sigma))

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd)/total:
                np.append(spoofed_dataset, box_blur(image, box_blur_kernel) )

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd)/total:
                np.append(spoofed_dataset, uniform_noise(image, uniform_max_noise))

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd + perlin_noise_odd)/total:
                np.append(spoofed_dataset, perlin_noise(image, perlin_max_noise))

            else:
                np.append(spoofed_dataset, flip_image(image))

    return spoofed_dataset

# Visualizing blur 

# fig = plt.figure(figsize=(5, 5))

# for i in range(5):
#     for j in range(5):
#         fig.add_subplot(i, j)
#         plt.imshow(gaussian_blur(x_train[j], i))
# plt.show()

# f, axes = plt.subplots(5, 5)
    
# for i, row in enumerate(axes):
#     for j, ax in enumerate(row):
#         ax.imshow(gaussian_blur(x_train[j], 2))

# plt.show()
