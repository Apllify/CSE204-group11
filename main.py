import matplotlib as plt
import numpy as np
from models import CNN_model, PCA_model
from img_manipulations import *
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import utils
from tensorflow.keras import models
import matplotlib.pyplot  as plt
from tests import *


MAX_BLUR = 2
#load dataset + normalize
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
x_train, y_train, x_test, y_test = prep_rotations(x_train, y_train, x_test, y_test)
x_train = x_train / 255  
x_test = x_test / 255

y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)


def one_hot_encode(y):
    res = np.zeros((y.size, 10))
    res[np.arange(y.size), y] = 1
    return res


""" PCA Model Training Code
pca_model = PCA_model(250, 10)
pca_model.fit(x_train, y_train_cat)
pca_model.save("pca_weights")
"""

""" DNN Model Training Code
dnn_model = PCA_model(28*28, 10, True, False)
dnn_model.fit(x_train, y_train_cat)
dnn_model.save("dnn_weights")
"""

""" CNN Model Training Code
cnn_model.fit(x_train.reshape(-1, 28, 28, 1), one_hot_encode(y_train))
cnn_model.save_weights('cnn_weights')
"""


#so clean :0
pca_model = PCA_model.load("pca_weights") 
dnn_model = PCA_model.load('dnn_weights') 

cnn_model = CNN_model()
cnn_model.load_weights('cnn_weights')





# #ROTATION TESTS
angles = np.arange(0, 360)
cnn_acc, pca_acc, dnn_acc = attack_tests(angles, rotate_database, x_test, y_test, cnn_model, pca_model, dnn_model)

#GAUSSIAN BLUR TESTS
blurs = np.arange(0, 2, 0.1)
cnn_acc, pca_acc, dnn_acc = attack_tests(blurs, gaussian_blur_database, x_test, y_test, cnn_model, pca_model, dnn_model)
