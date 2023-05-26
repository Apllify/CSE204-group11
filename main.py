import matplotlib as plt
import numpy as np
from models import CNN_model, PCA_model
from img_manipulations import *
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import utils
from tensorflow.keras import models
import matplotlib.pyplot  as plt

MAX_BLUR = 2
#load dataset + normalize
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
x_train = x_train / 255  
x_test = x_test / 255



def one_hot_encode(y):
    res = np.zeros((y.size, 10))
    res[np.arange(y.size), y] = 1
    return res



# y_train_cat = utils.to_categorical(y_train, 10)



pca_model = PCA_model.load("pca_weights") #250 component pca 
dnn_model = PCA_model.load_model(models.load_models('dnn_weights'), 28*28, 10, x_train) #this probably needs to be changed



cnn_model = CNN_model()
cnn_model.load_weights('cnn_weights')
# cnn_model.fit(x_train.reshape(-1, 28, 28, 1), one_hot_encode(y_train))
# cnn_model.save_weights('cnn_weights')
# loss, acc = cnn_model.evaluate(x_test.reshape(-1, 28, 28, 1), one_hot_encode(y_test))
# print(loss, acc)


#comment for testing


#ROTATION TESTS
angles = np.arange(0, 360)

cnn_acc = np.zeros(360)
pca_acc = np.zeros(360)
dnn_acc = np.zeros(360)

for a in enumerate(angles):
    cnn_acc[a] = cnn_model.evaluate(rotate_database(x_test, 0, a))[1]
    pca_acc[a] = pca_model.evaluate(rotate_database(x_test, 0, a))[1]
    dnn_acc[a] = dnn_model.evaluate(rotate_database(x_test, 0, a))[1]



#BLUR TESTS
blurs = np.arange(0, 360)

cnn_acc = np.zeros(360)
pca_acc = np.zeros(360)
dnn_acc = np.zeros(360)

for a in enumerate(angles):
    cnn_acc[a] = cnn_model.evaluate(rotate_database(x_test, 0, a))[1]
    pca_acc[a] = pca_model.evaluate(rotate_database(x_test, 0, a))[1]
    dnn_acc[a] = dnn_model.evaluate(rotate_database(x_test, 0, a))[1]
