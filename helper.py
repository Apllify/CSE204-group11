"""
PREFACE:
    To isntall perlin_numpy, run : "pip install git+https://github.com/pvigier/perlin-numpy"


SUMMARY : 
    This file contains the following image modification functions :
        
        - rotate_image(img, rot_angle)
        
        - gaussian_blur(img, sigma=1)
        
        - box_blur(img, kernel = 2)
        
        - uniform_noise(img, max_noise)
        
        - perlin_noise(img, max_noise)
"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

import tensorflow.keras.datasets.mnist as mnist
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

import scipy
import perlin_numpy 






"""
rotate_image(img, rot_angle)
img : an image in numpy array form
rot_angle : the angle of rotation in degrees (the rotation is counter-clockwise)

returns a new rotated version of the image
"""
def rotate_image(img, rot_angle):
    return  scipy.ndimage.rotate(img, rot_angle, reshape=False)


"""
gaussian_blur(img, sigma=1)
img : an image in numpy array form
sigma : the intensity of the blur, anything above 1.5 is very hard to recognize for humans

returns a new blurred version of the image
"""
def gaussian_blur(img, sigma=1):
    return scipy.ndimage.gaussian_filter(img, sigma)


"""
box_blur(img, kernel=2)
img : an image in numpy array form
kernel : the integer size of the box used in the blur, woudln't reccomend going above 4

returns a new blurred version of the image
"""
def box_blur(img, kernel=2):
    return scipy.ndimage.uniform_filter(img, kernel)


"""
uniform_noise(img, max_noise)
img : an image in numpy array form
max_noise : the maximum absolute value of the noise added to any given pixel, doesn't affect frequency

returns a new noisier version of the image
"""
def uniform_noise(img, max_noise):
    noise = (np.random.rand(*img.shape) - 0.5) * 2 * max_noise
    return img + noise



"""
perlin_noise(img, max_noise)
img : an image in numpy array form
max_noise : the maximum absolute value of the perlin noise added to any given pixel, doesn't affect frequency

returns a new noisier version of the image
"""
def perlin_noise(img, max_noise):
    noise = perlin_numpy.generate_perlin_noise_2d((28, 28), (2, 2)) * (max_noise * 2)
    
    return img + noise


"""
Invert colors of input image
"""
def flip_image(image):
    return 1 - image
