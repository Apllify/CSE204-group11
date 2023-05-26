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





#so clean :0
# pca_model = PCA_model.load("pca_weights") #250 component pca 
# #dnn_model = PCA_model.load_model(models.load_models('dnn_weights'), 28*28, 10, x_train) #this probably needs to be changed



cnn_model = CNN_model()
cnn_model.load_weights('cnn_weights')
# # cnn_model.fit(x_train.reshape(-1, 28, 28, 1), one_hot_encode(y_train))
# # cnn_model.save_weights('cnn_weights')
# # loss, acc = cnn_model.evaluate(x_test.reshape(-1, 28, 28, 1), one_hot_encode(y_test))
# # print(loss, acc)



#pca_model = PCA_model(50, 10)
#pca_model.fit(x_train[:100], y_train_cat[:100])
#pca_model.save("minimodel")


pca_model = PCA_model.load("minimodel")

# #ROTATION TESTS
angles = np.arange(0, 360)
cnn_acc, pca_acc, dnn_acc = run_tests(angles, rotate_database, cnn_model, pca_model, dnn_model)

# #GAUSSIAN BLUR TESTS
blurs = np.arange(0, 2, 0.1)
cnn_acc, pca_acc, dnn_acc = run_tests(blurs, gaussian_blur_database, cnn_model, pca_model, dnn_model)

# cnn_acc = np.zeros(len(blurs))
# pca_acc = np.zeros(len(blurs))
# dnn_acc = np.zeros(len(blurs))

# for b in enumerate(blurs):
#     cnn_acc[a] = cnn_model.evaluate(gaussian_blur_database(x_test, 0, b))[1]
#     pca_acc[a] = pca_model.evaluate(gaussian_blur_database(x_test, 0, b))[1]
#     #dnn_acc[a] = dnn_model.evaluate(gaussian_blur_database(x_test, 0, b))[1]
