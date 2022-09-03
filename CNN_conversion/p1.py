import numpy as np
import pandas as pd
import math as m
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import SGD, Adam



class convert:
    
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.split()
        self.scale_pca()
        self.CNN()
        
        
    def split(self):
        
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.data,
                                                                                              self.targets, test_size = 0.2)
        
        return self.train_data, self.test_data, self.train_label, self.test_label

    def img_convert(self, data, length, width): 
        
        final = np.zeros((len(data), length, width))
        for i in range(len(data)):
            new = np.reshape(data[i], (length, width))
            final[i] = new
        return final

    def lowest_square(self, x):
        while m.sqrt(x).is_integer() == False:
            x = x - 1
        return x   

    def scale_pca(self):
        
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data)
        train_data_scale = self.scaler.transform(self.train_data)
        test_data_scale  = self.scaler.transform(self.test_data)

        self.ls = self.lowest_square(self.data.shape[1])
        self.sqrt = int(m.sqrt(self.ls))

        self.pca = PCA(n_components = self.ls)
        self.pca.fit(train_data_scale)
        pcadata_train = self.pca.transform(train_data_scale)
        pcadata_test  = self.pca.transform(test_data_scale)
       
        pca_train =  self.img_convert(pcadata_train, self.sqrt, self.sqrt)
        pca_test  =  self.img_convert(pcadata_test , self.sqrt, self.sqrt)

        self.pca_train = np.expand_dims(pca_train, -1)
        self.pca_test  = np.expand_dims(pca_test, -1)
    
       
        return self.pca, self.scaler, self.ls, self.sqrt, self.pca_train, self.pca_test
    
    def CNN(self):
        
        monitor_val_acc = EarlyStopping(monitor = 'val_accuracy', patience = 10)
    
        self.model = Sequential()
        self.model.add(Conv2D(128, kernel_size = (int(np.round(m.sqrt(self.sqrt))), int(np.round(m.sqrt(self.sqrt)))),
                              input_shape = (self.sqrt, self.sqrt, 1), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(64, kernel_size = (int(np.round(m.sqrt(self.sqrt))) - 1, int(np.round(m.sqrt(self.sqrt))) - 1),
                              activation = 'relu'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        self.model.fit(self.pca_train, self.train_label, epochs = 100, validation_split = 0.1, callbacks = [monitor_val_acc])
        
        return self.model
    
    def evaluate(self):
        
        self.model.evaluate(self.pca_test, self.test_label)
        
    def scale_pca_test(self, test):

        test = self.scaler.transform(test)      

        self.ls = self.lowest_square(self.data.shape[1])
        self.sqrt = int(m.sqrt(self.ls))

        test = self.pca.transform(test)

        test =  self.img_convert(test, self.sqrt, self.sqrt)

        test = np.expand_dims(test, -1)

        return test
    
    def plot(self, data, x):
        plt.figure(figsize =(20, 20))
        for i in range(x):
            plt.subplot(m.ceil(m.sqrt(x)), m.ceil(m.sqrt(x)), i+1)
            try: 
                plt.imshow(data[i])
            except:
                plt.imshow(data[i].squeeze())
            plt.colorbar()
            plt.grid(False)
        plt.show()
        

    def conf_matrix(self):
        
        preds = np.round(self.model.predict(self.pca_test))

        class_names = np.array([' Dodgey', ' Legit'])

        cm = confusion_matrix(np.array(self.test_label), preds)

        plot_confusion_matrix(conf_mat = cm, class_names = class_names, colorbar = True)
        plot_confusion_matrix(conf_mat = cm, class_names = class_names, show_absolute = False,
                              show_normed = True, colorbar = True)
