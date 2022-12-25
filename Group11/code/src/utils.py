import numpy as np
import matplotlib.pyplot as plt

def make_preds(model,n,arr,time_step):
  c=[]
  curr=arr[-time_step:].copy()
  while(n>0):
    pred=model.predict(np.array([curr]))
    pred=np.array(pred)
    pred=np.reshape(pred,(pred.shape[0]))
    curr[0:time_step-1]=curr[1:time_step]
    curr[time_step-1]=pred
    c.append(pred)
    n=n-1
  return np.array(c)

#gives sequential output::
def process(arr,time_step):
  x=[]
  y=[]
  for i in range(arr.shape[0]-time_step):
     x.append(arr[i:i+time_step])
     y.append(arr[i+time_step])
  x=np.array(x,dtype='float32')
  y=np.array(y,dtype='float32')
  x=np.reshape(x,(x.shape[0],x.shape[1],1))
  return (x,y)

def plot(y_true,y_pred,fn='random'):
    print(y_pred.shape)
    type=fn.split('_')
    if(len(type)!=1):
      plt.title(type[1] + f" on algorithm {type[0]}")
    plt.ylabel('stock price')
    plt.xlabel('time->')
    plt.plot(range(y_true.size),y_pred,'r-',label='predicted stock value')
    plt.plot(range(y_true.size),y_true,'g-',label='actual stock value')
    plt.legend()
    plt.savefig(f"./reports/{fn}.png")

def sequential_data(x,type=0,scale=1):
    if(type==0):
      df=np.zeros((x.shape[0],1))
      #normal average of data::
      df[:,0]=(x['High']+x['Low'])/(2*scale)
      return df
    if(type==1):
      df=np.zeros((x.shape[0],2))
      #normal average of data::
      df[:,0]=(x['High']/(scale))
      df[:,1]=(x['Low']/(scale))
      return df
      #two dimensional oc input::

  

