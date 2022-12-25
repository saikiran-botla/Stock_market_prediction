from tensorflow.keras.layers import Dense,RNN,LSTM
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import tensorflow.keras
import pickle
from tensorflow.keras.layers import Dropout,BatchNormalization,Reshape
from utils import *
class nn:
    def __init__(self,lr,time_step):
        self.time_step=time_step
        self.model=Sequential()
        self.model.add(Reshape(target_shape=(time_step,),input_shape=(time_step,1)))
        self.model.add(Dense(120))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(8))
        self.model.add(Dense(1))
        #opt=tensorflow.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer='adam',loss="mean_squared_error")

    def save_model(self):
        self.model.save('./models/NN')

    def load_model(self):
        self.model=tensorflow.keras.models.load_model('./models/NN')

    def train(self,train_x,train_y,iters=100,batch_size=120,verbose=1):
        self.history=self.model.fit(train_x,train_y,epochs=iters,batch_size=batch_size,verbose=verbose)
        print('------------TRAINING DONE------------')

    def iter_pred(self,n,arr):
        #iteratively predicts multiple future values::
        return make_preds(self.model,n,arr,self.time_step)

    def make_pred(self,n,arr,test):
        #predcits only one step ahead::
            N=n
            c=[]
            curr=arr[-1].copy()
            print(curr.shape)
            while(n>0):
                pred=self.model.predict(np.array([curr]))
                pred=np.array(pred)
                pred=np.reshape(pred,(pred.shape[0]))
                curr[0:self.time_step-1]=curr[1:self.time_step]
                curr[self.time_step-1]=test[N-n]
                c.append(pred)
                n=n-1
            return np.array(c)

    def plot_history(self):
        #will use self.history to plot after training has been done::
        pass

    def predict(self,arr):
        return self.model.predict(arr)
