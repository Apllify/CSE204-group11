import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import SGD


class CNN_model(Model):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = Conv2D(64, 8, strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(128, 6, strides=(2, 2), activation="relu", padding="valid")
        self.conv3 = Conv2D(128, 5, strides=(1, 1), activation="relu", padding="valid")
        self.dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
    
    
class PCA_model(object):
    
    #constants
    batch_size = 32
    epochs = 5
    
    

    
    """
    init assumes that input is already flattened
    """
    def __init__(self, input_size, output_size, component_count):
        
        self.input_size = input_size
        self.output_size = output_size
        self.component_count = component_count
        
        #create network layer by layer
        self.input = Input(component_count)
        
        self.dense1 = Dense(2048, activation="relu")(self.input)
        self.dense2 = Dense(1024, activation="relu")(self.dense1)
        self.dense3 = Dense(512, activation="relu")(self.dense2)
        self.dense4 = Dense(256, activation="relu")(self.dense3)
        
        self.output= Dense(output_size, activation="linear")(self.dense4)
        
        
        #compile network 
        sgd = SGD(learning_rate=0.1)
        self.output.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        
    def PCA_truncate(self, data, component_count):
        #todo : pca truncation 
        pass 
    
    def fit(self, x, y):

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, mode='auto')
        truncated_data = self.PCA_truncate(x)
        
        self.out.fit(truncated_data, y, batch_size=PCA_model.batch_size, epochs=PCA_model.epochs, 
                   shuffle=True, callbacks=[early_stopping])
        