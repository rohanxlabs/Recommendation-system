import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD

def train_model(matrix):
    n_components = min(matrix.shape) - 1
    svd = TruncatedSVD(n_components=n_components,random_state=42)
    latent_matrix = svd.fit_transform(matrix)
    return svd,latent_matrix

def save_model(model,path):
    with open(path,'wb') as f:
        pickle.dump(model,f)