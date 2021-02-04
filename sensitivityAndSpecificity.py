import numpy as np

def sensitivityAndSpecificity(y_pred, y_true):
  sens = sum((y_pred==1)&(y_true==1))/sum(y_true==1)
  neg = sum((y_pred==0)&(y_true==0))/sum(y_true==0)

  return [sens, neg]


sensitivityAndSpecificity(np.array([0,1,1,0,1,1,0,1,0,0,1]),np.array([1,1,1,0,0,0,1,1,0,1,1]))

