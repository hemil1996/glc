import numpy as np
import pandas as pd
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import keras
from tqdm import tqdm
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Dropout,Activation
from tensorflow.python.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = # Provide path for dataset 
X =   # features
y =  # labels

x = (X - X.mean())/X.std()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
features = x_train.shape[1]
classes = len(np.unique(y_train))

model = Sequential()
model.add(Dense(units=16,activation='relu',input_dim = features))
model.add(Dropout(0.2))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units= classes, activation='softmax'))  
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
es = [EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode = 'auto'),
      ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')]


history = model.fit(x_train, y_train,epochs=10,callbacks=es,validation_split=0.33)

c_m = np.zeros((classes,classes))
for i in tqdm(range(0,len(x_train))):
  row_no = (model.predict(x_train[i:i+1])[0]).argmax()
  c_m[row_no-1,:] +=model.predict(x_train[i:i+1])[0]
c_matrix = np.divide(c_m,len(x_train))    # divide each line with data length

model2 = Sequential()
model2.add(Dense(units=16,activation='relu',input_dim = features))
#model.add(Dropout(0.2))
model2.add(Dense(units=32,activation='relu'))
#model.add(Dropout(0.2))
model2.add(Dense(units= classes, activation='softmax'))  
model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
es = [EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode = 'auto'),
      ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')]

history2 = model2.fit(x_train, y_train, batch_size=5,epochs=10,callbacks=es,validation_split=0.33)
c_matrix = c_matrix.transpose()


y_new = np.zeros((len(x_train),1))
for k in tqdm(range(0,len(x_train))):
  y_new[k,:] = (np.matmul(c_matrix,(history2.model.predict(x_train[k:k+1])[0]).reshape((classes,1)))).argmax()


classes = len(np.unique(y_new))  # to see all classes 

history2 = model.fit(x_train, y_new, batch_size=5,epochs=50,shuffle=True,callbacks=es,validation_split=0.33)

history2.model.evaluate(x_test,y_test)      