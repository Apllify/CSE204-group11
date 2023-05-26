"""
PREFACE:
    To isntall perlin_numpy, run : "pip install git+https://github.com/pvigier/perlin-numpy"


SUMMARY : 
    This file contains the following image modification functions :
        - flip_image(img) 
        
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


def rotate_image(img, rot_angle):
    """
    rotate_image(img, rot_angle)
    img : an image in numpy array form
    rot_angle : the angle of rotation in degrees (the rotation is counter-clockwise)

    returns a new rotated version of the image
    """
    return scipy.ndimage.rotate(img, rot_angle, reshape=False)


def gaussian_blur(img, sigma=1):
    """
    gaussian_blur(img, sigma=1)
    img : an image in numpy array form
    sigma : the intensity of the blur, anything above 1.5 is very hard to recognize for humans

    returns a new blurred version of the image
    """
    return scipy.ndimage.gaussian_filter(img, sigma)


def box_blur(img, kernel=2):
    """
    box_blur(img, kernel=2)
    img : an image in numpy array form
    kernel : the integer size of the box used in the blur, woudln't reccomend going above 4

    returns a new blurred version of the image
    """
    return scipy.ndimage.uniform_filter(img, kernel)


def uniform_noise(img, max_noise):
    """
    uniform_noise(img, max_noise)
    img : an image in numpy array form
    max_noise : the maximum absolute value of the noise added to any given pixel, doesn't affect frequency

    returns a new noisier version of the image
    """
    noise = (np.random.rand(*img.shape) - 0.5) * 2 * max_noise
    return img + noise


def perlin_noise(img, max_noise):
    """
    perlin_noise(img, max_noise)
    img : an image in numpy array form
    max_noise : the maximum absolute value of the perlin noise added to any given pixel, doesn't affect frequency

    returns a new noisier version of the image
    """
    noise = perlin_numpy.generate_perlin_noise_2d(
        (28, 28), (2, 2)) * (max_noise * 2)

    return img + noise


def flip_image(image):
    """
    Invert colors of input image
    """
    return 1 - image



#def rotate_database(image, )