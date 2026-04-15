import pandas as pd 
from sklearn.model_selection import train_test_split

def preprocess_data(path):
    df = pd.read_csv(path)
    train,test = train_test_split(df, test_size=0.2,random_state=42)
    train.to_csv("data/processed/train.csv",index=False)
    test.to_csv("data/processed/test.csv",index=False)
    return train,test