import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


#data prepration
def Normalize_data(data):
    min_a = np.min(data, axis=0, keepdims=True)
    max_a = np.max(data, axis=0, keepdims=True)
    data=np.divide((data-min_a), (max_a-min_a))
    return data
def Standardize_data(data):
    mean_a = np.mean(data, axis=0, keepdims=True)
    std_a = np.std(data, axis=0, keepdims=True)
    data=np.divide((data-mean_a), std_a)
    return data


    
def None_initialaizer(param, nlayers, value):
    if param is None:
        param=[value]*nlayers  
    return param   
    

# takes arrays of length (input layer + N hidden layer + output layer). The default values are specified in the funciton    
def generate_model_DNN(units, activation=None, use_bias=None,
    kernel_initializer=None,
    bias_initializer=None, kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, drop_prob=0.001, use_dropout = None):
    nlayers=len(units)
    activation = None_initialaizer(activation, nlayers, None)
    use_bias = None_initialaizer(use_bias, nlayers, False)
    use_dropout = None_initialaizer(use_dropout, nlayers, False)
    kernel_regularizer = None_initialaizer(kernel_regularizer, nlayers, None)
    kernel_initializer = None_initialaizer(kernel_initializer, nlayers, 'glorot_uniform')
    bias_initializer = None_initialaizer(bias_initializer, nlayers, 'zeros')
    activity_regularizer = None_initialaizer(activity_regularizer, nlayers, None)
    kernel_constraint = None_initialaizer(kernel_constraint, nlayers, None)
    bias_constraint = None_initialaizer(bias_constraint, nlayers, None)
    bias_regularizer = None_initialaizer(bias_regularizer, nlayers, None)

    DNN = Sequential()
    for i in range(nlayers):
        DNN.add(
            Dense(
    units[i], activation = activation[i], use_bias = use_bias[i],
    kernel_initializer = kernel_initializer[i],
    bias_initializer = bias_initializer[i], kernel_regularizer = kernel_regularizer[i],
    bias_regularizer = bias_regularizer[i], activity_regularizer = activity_regularizer[i], kernel_constraint = kernel_constraint[i],
    bias_constraint = bias_constraint[i])
        )
    if use_dropout == True:
        DNN.add(Dropout(drop_prob))
       

    return DNN




#spliting data to train, val and test. p_vt=fraction of validation and test set (defaul=0.2). p_t=fraction of test set from test_validation set (default=0.5). 

def split_tvt(X, y, p_vt=0.2, p_t=0.5):
    X_split, y_split = {}, {}
    X_split["train"], X_split["test"], y_split["train"], y_split["test"] = train_test_split(X, y, test_size=0.20, random_state=42)
    X_split["val"], X_split["test"], y_split["val"], y_split["test"]= train_test_split(X_split["test"], y_split["test"], test_size=0.5, random_state=42)
    return X_split, y_split








