import os
import sys
import pandas as pd
from config import *

sys.path.append('./src')
from LSTM import lstm
from NN import nn
from GRU import gru
from MA import ma
from utils import *
#take in arguments to train/run::
type=sys.argv[1]
model=sys.argv[2]

#config has all training and other configuration variables stored::
time_step=config['step_time']

if(type=='train'):
    raw_train=pd.read_csv('./data/training.csv')
    train=sequential_data(raw_train,0,10)
    train_x,train_y=process(train,time_step)
    if(model=='LSTM'):
        mod=lstm(config['train']['LSTM']['lr'],time_step)
        iters=config['train']['LSTM']['iters']
    elif(model=='NN'):
        mod=nn(config['train']['NN']['lr'],time_step)
        iters=config['train']['NN']['iters']
    elif(model=='GRU'):
        mod=gru(config['train']['GRU']['lr'],time_step)
        iters=config['train']['GRU']['iters']
    elif(model=='MA'):
        mod=ma(time_step)
        iters=0

    #training begins::
    mod.train(train_x,train_y,iters=iters,batch_size=time_step)
    mod.save_model()
    #plotting relevent quantities::
    mod.plot_history()
    #plotting model performance::
    plot(train_y,mod.predict(train_x),model+'_fit')

        #getting test_set

else:
    #this is where testing will be done:: 
    #graph generations also should be done from here::
    raw_train=pd.read_csv('./data/training.csv')
    raw_test= pd.read_csv('./data/test.csv')

    if(model=='LSTM'):
        mod=lstm(config['train']['LSTM']['lr'],time_step)
    elif(model=='NN'):
        mod=nn(config['train']['NN']['lr'],time_step)
    elif(model=='GRU'):
        mod=gru(config['train']['GRU']['lr'],time_step)
    elif(model=='MA'):
        mod=ma(time_step)
        
    #loading model::
    mod.load_model()
    
    train=sequential_data(raw_train,0,10)
    train_x,train_y=process(train,time_step)
    test=sequential_data(raw_test,0,10)

    y_pred=mod.make_pred(test.shape[0],train_x,test)
    
    print(f"COEFFICIENT OF VARIANCE: {(np.linalg.norm(y_pred-test,2)/np.sqrt(test.size))/np.mean(test)}")
    #plotting model performance::
    plot(y_pred,test,model+'_test')


