import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, BatchNormalization
from tensorflow.keras.optimizers import SGD

import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt


class CNN_model(Model):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), strides=(
            2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(128, (2, 2), strides=(
            1, 1), activation="relu", padding="valid")
        self.conv3 = Conv2D(128, (3, 3), strides=(
            1, 1), activation="relu", padding="valid")
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
        self.compile(optimizer=SGD(),
                     loss='categorical_crossentropy', metrics=['accuracy'])


class PCA_model(object):

    # constants
    batch_size = 32
    epochs = 5


    def __init__(self, component_count, output_size, generate_network=True, perform_PCM=True):
        
        """
        init sets up the model itself
        """

        # instance attributes
        self.output_size = output_size
        self.component_count = component_count
        self.perform_PCM = perform_PCM
            
        
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
        flat_array = flat_array.reshape(
            flat_array.shape[0], np.prod(flat_array.shape[1:]))

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

        

    def fit(self, x, y):
        """
        Truncates all of the inputs
        Trains on that new version of them to reduce density of the input layer 
        """
            
        
        if (not self.is_trained):
            if (self.perform_PCM):
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

        if (self.perform_PCM):
            pca_x = np.matmul(flat_x, self.truncate_matrix.T[:, :self.component_count])
            return self.model.predict(pca_x)
        else:
            return self.model.predict(flat_x)
        
    
    
    def save(self, filename):
        self.model.save(filename)