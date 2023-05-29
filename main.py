#global imports
import matplotlib as plt
import numpy as np
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import utils
from tensorflow.keras import models
import matplotlib.pyplot  as plt

#local imports
from models import CNN_model, PCA_model
from attack_tests import run_attacks, generate_spoofed_dataset
from img_manipulations import *
from database_manipulations import *


#CONSTANTS
MAX_BLUR = 2



#Load the main MNIST Dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
x_train = x_train / 255  
x_test = x_test / 255

#get categorical versions of the labels
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

#special version of dataset for rotations
rx_train, ry_train = prep_rotations(x_train, y_train)
rx_test, ry_test = prep_rotations(x_test, y_test)

#get categorical versions of the rotation-ready labels
ry_train_cat = utils.to_categorical(ry_train, 10)
ry_test_cat = utils.to_categorical(ry_test, 10)


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
cnn_model.fit(x_train.reshape(-1, 28, 28, 1), y_train_cat)
cnn_model.save_weights('cnn_weights')
"""


#loading the models from memory
pca_model = PCA_model.load("pca_weights") 
dnn_model = PCA_model.load("dnn_weights") 

cnn_model = CNN_model()
cnn_model.load_weights('cnn_weights')



#BOILERPLATE code for generating and plotting the effect of an attack
n_samples = 12
attack_function = box_blur_database

arguments = np.zeros((n_samples, 2))
arguments[:, 0] = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
arguments[:, 1] = arguments[:, 0]

x_axis = arguments[:, 0]

model_list = [pca_model, dnn_model, cnn_model]


result = run_attacks(rx_test, ry_test_cat, model_list, attack_function, arguments)


plt.plot(x_axis, result[0], label="PCA Model")
plt.plot(x_axis, result[1], label="DNN Model")
plt.plot(x_axis, result[2], label="CNN Model")
plt.legend(loc="upper right")
plt.xlabel("Kernel/Intensity of Box Blur")
plt.ylabel("Accuracy of models")


plt.show()



# fig = plt.figure()
# noise_test = x_test[0]

# for i in range(25):
#     fig.add_subplot(5, 5, i+1)
#     plt.axis("off")
#     plt.imshow(flip_image(x_test[i]))
