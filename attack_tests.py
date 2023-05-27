import numpy as np
import random
import warnings

from img_manipulations import *

warnings.filterwarnings('ignore')


def run_attacks(database_x, database_y, model_list, attack_func, attack_arguments) :
    
    '''
    
    Run attack and return accuracy lists for all three models
    
    database_x : the x database, in image (2d) form
    database_y : the y database, MUST be categorical
    
    model_list : the list of models to be tested, they MUST all have a .evaluate() method
    attack_func : function that takes a database and returns a modified version
    
    attack_arguments : a list of the arguments that the attack function will take
    EXAMPLE : if attack_func takes 3 arguments (database included) then we can have : 
        attack_arguments = [ (0, 1), (2, 8), (0, 3), etc... ]
        
        
        
    Returns a matrix with N rows and M columns
    where N : number of models in model_list
    and M : number of separate arguments in attack_arguments
    
    IF there is only one model given, the function returns a simple list of all
    the accuracy samples
        
    '''
    
    n_measures = len(attack_arguments)
    n_models = len(model_list)
    
    accs = np.zeros((n_models, n_measures)) #one row per model



    for measure_i in range(n_measures):
        
        cur_arguments = attack_arguments[measure_i]
        new_database_x = attack_func(database_x, *cur_arguments)
        
        for model_i in range(n_models):
            
            current_acc = model_list[model_i].evaluate(new_database_x, database_y)[1]
            accs[model_i, measure_i] = current_acc

    

    #if only one model was given, simplify output
    if accs.shape[0] == 1:
        accs = accs[0]
    
    
    return accs



def generate_spoofed_dataset(database_x, database_y):
    """
    Generates a list of same length and labels as x but with random filter functions
    applied to each element

    database_y should NOT be categorical, just the regular database_y
    
    """
    spoofed_dataset = np.zeros_like(database_x)

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
                spoofed_dataset[i] = rotate_image(image, rotation_angle)

            elif rand <= (rotation_odd + gaussian_blur_odd)/total:
                spoofed_dataset[i] = gaussian_blur(image, gaussian_blur_sigma)

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd)/total:
                spoofed_dataset[i] = box_blur(image, box_blur_kernel) 

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd)/total:
                spoofed_dataset[i] = uniform_noise(image, uniform_max_noise)

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd + perlin_noise_odd)/total:
                spoofed_dataset[i] = perlin_noise(image, perlin_max_noise)

            else:
                spoofed_dataset[i] = flip_image(image)


        else:
            if rand <= (rotation_odd + gaussian_blur_odd)/total:
                spoofed_dataset[i] = gaussian_blur(image, gaussian_blur_sigma)

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd)/total:
                spoofed_dataset[i] = box_blur(image, box_blur_kernel) 

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd)/total:
                spoofed_dataset[i] = uniform_noise(image, uniform_max_noise)

            elif random <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd + perlin_noise_odd)/total:
                spoofed_dataset[i] = perlin_noise(image, perlin_max_noise)

            else:
                spoofed_dataset[i] = flip_image(image)

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
