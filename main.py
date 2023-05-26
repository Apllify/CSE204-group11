import numpy as np
from models import CNN_model, PCA_model
import img_manipulations
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras import utils
from tensorflow.keras import models


#load dataset + normalize
(x_train, y_train), (x_test, y_test) = mnist.load_data()  
x_train = x_train / 255  
x_test = x_test / 255

def one_hot_encode(y):
    res = np.zeros((y.size, 10))
    res[np.arange(y.size), y] = 1
    return res



# y_train_cat = utils.to_categorical(y_train, 10)
# pca_model = PCA_model(50, 10)
# pca_model.fit(x_train[:1000], y_train_cat[:1000])


# pca_model.save('pca_weights')
pca_model = models.load_model('pca_weights')



cnn_model = CNN_model()
cnn_model.load_weights('cnn_weights')
# cnn_model.fit(x_train.reshape(-1, 28, 28, 1), one_hot_encode(y_train))
# cnn_model.save_weights('cnn_weights')
# loss, acc = cnn_model.evaluate(x_test.reshape(-1, 28, 28, 1), one_hot_encode(y_test))
# print(loss, acc)




