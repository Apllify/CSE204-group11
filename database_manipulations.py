#global imports
import numpy as np
import pandas as pd
import random

#local imports
from img_manipulations import *
 

def prep_rotations(x, y):
    """
    Removes all instances of the numbers 6 and 9 from input datasets
    
    Returns the new, purged, datasets
    """

    new_x = []
    new_y = []


    for i, num in enumerate(y):
        if num not in (6, 9):
            new_y.append(num)
            new_x.append(x[i])
    


    return (np.array(new_x), np.array(new_y))


def rotate_database(images, min_rot, max_rot):
    """
    Returns a new database that maches the rotation requirements

    min_rot, max_rot : rotation angles in degree, can be negative, must be integers
    """

    new_images = np.zeros_like(images)


    for i in range(images.shape[0]):
        rot = random.randint(min_rot, max_rot)
        new_images[i] = rotate_image(images[i], rot)

    return new_images



def gaussian_blur_database(images, min_blur, max_blur):
    """
    Returns a new database that maches the blur requirements

    For reference, blur of 1 is still legible, blur of >2 is inconsistently readable for humans
    """

    new_images = np.zeros_like(images)


    for i in range(images.shape[0]):
        rand= random.random()
        blur = rand*(max_blur - min_blur) + min_blur 
        new_images[i]= gaussian_blur(images[i], blur)

    return new_images



def box_blur_database(images, min_blur, max_blur):
    """
    Returns a new database that maches the blur requirements

    Average ok blur = 2
    """
    
    new_images = np.zeros_like(images)


    for i in range(images.shape[0]):
        rand= random.random()
        blur = rand*(max_blur - min_blur) + min_blur 
        new_images[i]= box_blur(images[i], blur)

    return new_images


def uniform_noise_database(images, max_noise):
    
    new_images = np.zeros_like(images)
    
    for i in range(images.shape[0]):
        new_images[i] = uniform_noise(images[i], max_noise)
        
    return new_images
        
        
def perlin_noise_database(images, max_noise):
    
    new_images = np.zeros_like(images)
    
    for i in range(images.shape[0]):
        new_images[i] = perlin_noise(images[i], max_noise)
        
    return new_images
    


def flip_database(images):
    
    new_images = 1-images
    
    return new_images
    
    
