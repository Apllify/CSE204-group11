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
    Basically, each row of the output is a series of accuracy samples of one of the models
    
    IF there is only one model given, the function returns a simple list of all
    the accuracy samples
        
    '''
    
    n_measures = len(attack_arguments)
    n_models = len(model_list)
    
    accs = np.zeros((n_models, n_measures)) #one row per model



    for measure_i in range(n_measures):
        
        #use the current arguments to generate a new, harder database
        try : 
            cur_arguments = attack_arguments[measure_i]
            new_database_x = attack_func(database_x, *cur_arguments)
        except:
            raise Exception ("Arguments count or shape didn't match function !")
        
        #measure each model's accuracy on that database
        for model_i in range(n_models):
            current_acc = model_list[model_i].evaluate(new_database_x, database_y)[1]
            accs[model_i, measure_i] = current_acc

    

    #if only one model was given, simplify output
    if accs.shape[0] == 1:
        accs = accs[0]
    
    
    return accs


def attack_lattice(model_class, train_database, test_database, attack_test, attack_train, range_test, range_train):
    '''
    Computes the lattice graph for the attack. 
    attack_func should be one of the <attack>_database functions, it is passed
    the elements of attack_range as argument for the attack intensity.
    '''
    
    lattice = np.zeros(shape=(len(range_train),len(range_test)))
    
    for i, x_I in np.ndenumerate(range_train):
        new_train_dat = attack_train(train_database[0], 0, x_I)
        model = model_class()
        model.fit(new_train_dat, train_database[1])
        for j, y_I in np.ndenumerate(range_test):
            new_test_dat = attack_test(test_database[0], 0, y_I)
            lattice[i][j] = model.evaluate(new_test_dat, test_database[1])[1]
            
    return lattice.T

def compute_average_confidence_over_right_answers(model, x, y):
    """computes the average confidence over images that were classified accurately. 
    y MUST NOT be one hot encoded"""
    y_pred = model.predict(x)
    correct_count = 0
    total_confidence = 0
    for i in range(y.shape[0]):
        if np.argmax(y_pred[i]) == y[i]:
            correct_count += 1
            total_confidence += np.maximum(y_pred[i])
    return total_confidence / correct_count
    
def compute_average_confidence_over_wrong_answers(model, x, y):
    """computes the average confidence in the wrong answers over images that were misclassified. 
    y MUST NOT be one hot encoded"""
    y_pred = model.predict(x)
    incorrect_count = 0
    total_confidence = 0
    for i in range(y.shape[0]):
        if np.argmax(y_pred[i]) != y[i]:
            incorrect_count += 1
            total_confidence += np.maximum(y_pred[i])
    return total_confidence / incorrect_count

def compute_average_confidence_over_true_right_answer(model, x, y):
    """computes average confidence in the right answer regardless of prediction.
    y MUST NOT be one hot encoded"""
    y_pred = model.predict(x)
    
    total_confidence = 0
    for i in range(y.shape[0]):
        total_confidence += y_pred[i][y[i]]
    return total_confidence / y.shape[0]

def generate_spoofed_dataset(database_x, database_y):
    """
    Generates a list of same length and labels as x but with random filter functions
    applied to each element

    database_y should NOT be categorical, just the regular database_y
    
    """
    spoofed_dataset = np.zeros_like(database_x)

    #FILTER PROBABILITIES (can be tweaked)
    rotation_odd = 2
    
    gaussian_blur_odd = 1
    box_blur_odd = 1
    
    uniform_noise_odd = 1
    perlin_noise_odd = 1
    
    flip_image_odd = 2
    

    #FILTER INTENSITIES (can be tweaked)
    rotation_min = 20
    rotation_max = 80
    
    gaussian_blur_sigma = 1
    box_blur_kernel = 2
    
    uniform_max_noise = 0.3
    perlin_max_noise = 0.3
    
    

    total = rotation_odd + gaussian_blur_odd + box_blur_odd + \
            uniform_noise_odd + perlin_noise_odd + flip_image_odd



    for i in range(database_x.shape[0]):
        
        rand = random.random() 
        image = database_x[i]

        if database_y[i] not in (6, 9): #avoid rotating the numbers 6 and 9 

            if rand <= (rotation_odd/total) : 
                
                #give rotation random angle and sign
                current_rot = (random.random() * (rotation_max - rotation_min)) + rotation_min
                random_sign = (random.randint(0, 1) * 2) - 1
                current_rot *= random_sign
                
                spoofed_dataset[i] = rotate_image(image, current_rot)

            elif rand <= ((rotation_odd + gaussian_blur_odd)/total):
                spoofed_dataset[i] = gaussian_blur(image, gaussian_blur_sigma)

            elif rand <= ((rotation_odd + gaussian_blur_odd + box_blur_odd)/total):
                spoofed_dataset[i] = box_blur(image, box_blur_kernel) 

            elif rand <= ((rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd)/total):
                spoofed_dataset[i] = uniform_noise(image, uniform_max_noise)

            elif rand <= ((rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd + perlin_noise_odd)/total):
                spoofed_dataset[i] = perlin_noise(image, perlin_max_noise)

            else:
                spoofed_dataset[i] = flip_image(image)


        else:
            if rand <= (rotation_odd + gaussian_blur_odd)/total:
                spoofed_dataset[i] = gaussian_blur(image, gaussian_blur_sigma)

            elif rand <= (rotation_odd + gaussian_blur_odd + box_blur_odd)/total:
                spoofed_dataset[i] = box_blur(image, box_blur_kernel) 

            elif rand <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd)/total:
                spoofed_dataset[i] = uniform_noise(image, uniform_max_noise)

            elif rand <= (rotation_odd + gaussian_blur_odd + box_blur_odd + uniform_noise_odd + perlin_noise_odd)/total:
                spoofed_dataset[i] = perlin_noise(image, perlin_max_noise)

            else:
                spoofed_dataset[i] = flip_image(image)

    return spoofed_dataset

