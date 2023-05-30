import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import models
from tensorflow.python.framework.ops import enable_eager_execution


import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')




class CNN_model(Model):
    
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(128, (2, 2), strides=(1, 1), activation="relu", padding="valid")
        self.conv3 = Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="valid")
        self.dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(10, activation="softmax")
        self.compile_SGD()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    def compile_SGD(self):
        self.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        
    def evaluate(self, x, y):
        return super().evaluate(x.reshape(-1, 28, 28, 1), y)
    
    def fit(self, x, y):
        return super().fit(x.reshape(-1, 28, 28, 1), y, epochs=3)
    
    def predict(self, x):
        return super().predict(x.reshape(-1, 28, 28, 1))
        



class PCA_model(object):

    # constants
    batch_size = 32
    epochs = 5


    def __init__(self, component_count=250, output_size=10, generate_network=True, perform_PCA=True):
        
        """
        init sets up the model itself
        """

        # instance attributes
        self.output_size = output_size
        self.component_count = component_count
        self.perform_PCA = perform_PCA
            
        
        self.is_trained = False
        
        if (generate_network):
            #create network layer by layer
            self.input = Input(component_count)
            
            self.dense1 = Dense(2048, activation="relu")(self.input)
            self.dense1 = BatchNormalization()(self.dense1)
            
            self.dense2 = Dense(1024, activation="relu")(self.dense1)
            self.dense2 = BatchNormalization()(self.dense2)
            self.dense3 = Dense(512, activation="relu")(self.dense2)
            self.dense3 = BatchNormalization()(self.dense3)

            self.dense4 = Dense(256, activation="relu")(self.dense3)
            self.dense4 = BatchNormalization()(self.dense4)
            
            self.output= Dense(output_size, activation="softmax")(self.dense4)
        
            #define model + compile
            self.model = Model(self.input, self.output, name="PCApredictor")
            
            #compile network 
            sgd = SGD(learning_rate=0.1)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
    def flatten_each_element(array):
        """
        Makes is so that every element of main array is a 1d array
        """

        if (len(array.shape) <= 2):
            return array

        flat_array = array[:]
        try:
            flat_array = flat_array.reshape(
                flat_array.shape[0], np.prod(flat_array.shape[1:]))
        except AttributeError:
            # enable_eager_execution()
            # print(tf.executing_eagerly())
            # flat_array = flat_array.numpy().reshape(flat_array.shape[0], np.prod(flat_array.shape[1:]))
            # flat_array = tf.convert_to_tensor(flat_array, float)
            tf.reshape(flat_array, (flat_array.shape[0], np.prod(flat_array.shape[1:])))

        return flat_array

    def PCA_truncate(self, data):
        """
        Computes and stores the W matrix during the training process
        """

        # flatten input data
        flat_data = PCA_model.flatten_each_element(data)

        # get pca coefficients
        u, sigma, w_transpose = np.linalg.svd(flat_data, full_matrices=False)

        # remember our w
        self.truncate_matrix = w_transpose

        # truncate data
        trunc_data = np.matmul(
            flat_data, w_transpose.T[:, :self.component_count])
        return trunc_data
        
            
    
    def load_model(loaded_model, component_count, output_size, data_set):
        """
        Loads model from preexisting weights 
        """
        pca = PCA_model(component_count, output_size, False)
        pca.model = loaded_model
        pca.PCA_truncate(data_set)

        return pca

    def __call__(self, x):
        """only in DNN mode"""
        if self.perform_PCA:
            raise Exception("Don't call in PCA mode")
        return self.model(x)

    def fit(self, x, y):
        """
        Truncates all of the inputs
        Trains on that new version of them to reduce density of the input layer 
        """
            
        
        if (not self.is_trained):
            if (self.perform_PCA):
                truncated_data = self.PCA_truncate(x)
            else:
                truncated_data = PCA_model.flatten_each_element(x)
            
            self.model.fit(truncated_data, y, batch_size=PCA_model.batch_size, epochs=PCA_model.epochs, 
                    shuffle=True)
            
            self.is_trained = True
      
        
    def predict(self, x):
        """
        Returns model predictions
        x has to be a LIST of inputs and not a singular input
        """

        if (not self.is_trained):
            raise Exception("Model hasn't been trained yet")

        if (len(x.shape) <= 1):
            raise Exception(
                "Input list of elements is invalid (dimension should be >= 2)")

        # hacky code, rework eventually
        flat_x = PCA_model.flatten_each_element(x)

        if (self.perform_PCA):
            pca_x = np.matmul(flat_x, self.truncate_matrix.T[:, :self.component_count])
            return self.model.predict(pca_x)
        else:
            return self.model.predict(flat_x)
        
    
    def evaluate(self, x_test, y_test):
        """
        Returns a tuple of the form (test loss, test acc)
        """

        if (not self.is_trained):
            raise Exception("Model hasn't been trained yet")


        x_test_flat= PCA_model.flatten_each_element(x_test)

        if (self.perform_PCA):
            pca_x = np.matmul(x_test_flat, self.truncate_matrix.T[:, :self.component_count])
            return self.model.evaluate(pca_x, y_test)
        else:
            return self.model.evaluate(x_test_flat, y_test)

    
    def save(self, foldername):
        """
        Saves network weights into given foldername
        """

        #model weights
        self.model.save(foldername)

        #also save transform matrix
        if (self.perform_PCA):
            np.save(f"{foldername}/PCAmatrix", self.truncate_matrix)
        else:
            np.save(f"{foldername}/PCAmatrix", np.array([]))







    def load(foldername):

        #load network + transf matrix
        network = models.load_model(foldername)
        truncate_matrix = np.load(f"{foldername}/PCAmatrix.npy")

        pca = PCA_model(network.input.shape[1], network.output.shape[1],False )
        pca.model = network
        pca.is_trained = True
        pca.truncate_matrix = truncate_matrix
        
        if(truncate_matrix.size == 0):
            pca.perform_PCA = False

        return pca

