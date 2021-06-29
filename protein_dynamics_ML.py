"""
NN model for prediction of the protein dynamics
@HamedEmamy
Jun 18 2021
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from datetime import datetime
from pathlib import Path
import functions
import sklearn
from sklearn import datasets


X, y = sklearn.datasets.make_regression(n_samples=1000, n_features=2, n_informative=2, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.05, shuffle=True, coef=False, random_state=None)

#X = np.load("idp_10260/idp_scd.npy")
#X = np.expand_dims(X, axis=1)
#X = np.append(X, np.expand_dims(np.load("idp_10260/idp_shd.npy"), axis=1), axis=1)
#y = np.load("idp_10260/idp_rg.npy")
y = np.expand_dims(y, axis=1)

listname=["train", "val", "test"]
X_split, y_split = functions.split_tvt(X, y)



#generate and train the NN
units = [X.shape[1], 5, 5,  y.shape[1]]
activation = ["linear", "relu", "relu", "linear"]
learning_rate = 0.001
callback = EarlyStopping(monitor='val_loss', patience=50)

model = functions.generate_model_DNN(units, activation=activation)
optimizer = keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X_split["train"], y_split["train"], batch_size=50, epochs=1000, validation_data=(X_split["val"], y_split["val"]), callbacks=[callback])



loss_train = model.history.history["loss"]
loss_val = model.history.history["val_loss"]
loss_test = np.mean(np.square((model.predict(X_split["test"])-y_split["test"])))


#plotting the losses
plt.plot(loss_train, label="training set loss")
plt.plot(loss_val, label="validation set loss")
plt.plot(len(loss_train)-1, loss_test, marker="o", label="test loss")



# save the model
dirname = "DNN_model"
Path(dirname).mkdir(parents=True, exist_ok=True)
model.save(dirname + "/DNN_model.h5")





