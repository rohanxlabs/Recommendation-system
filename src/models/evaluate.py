import numpy as np

def rmse(true,pred):
    return np.sqrt(np.mean((true - pred) **2))