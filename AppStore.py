# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:00:05 2018

@author: Mathew
"""

# Importing the libraries
import numpy as np
from scipy import sparse
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from matplotlib import style

# Importing the dataset
dataset = pd.read_csv('AppleStore.csv')
X = dataset.iloc[:, 3].values
y = dataset.iloc[:, 6].values

print(X)
print(y)

#displaying list of keys 

print("Keys of Apple stroe dataset: \n{}".format(dataset.keys()))


## Splitting the data into test and train data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


#Displaying shape of above mentioned arrays

print("X_train shape : {}".format(X_train.shape))
print("X_test shape : {}".format(X_test.shape))
print("y_train shape : {}".format(y_train.shape))
print("y_test shape : {}".format(y_test.shape))

## Visualizing Data

plt.plot(y,X)
plt.xlabel('Size in bytes')
plt.ylabel('Ratings')
plt.title('Ratings vs Size')
plt.show()



