import pickle
import numpy as np
import pandas as pd
import json

from sklearn import tree



class ModelPrediction:
    
    def __init__(self, path_dataset):
        with open("./src/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        df_test = pd.read_csv("./data/test.csv")
        
    def predict(self, X):
        return self.model.predict(X)

def main():

	trainer =  ModelPrediction("./data")
	with open("./tests/test_0.json") as f:
	    d = json.load(f)
	    predictions = trainer.predict(d["X"])
	    print(predictions, d["y"])
	with open("./tests/test_1.json") as f:
	    d = json.load(f)
	    predictions = trainer.predict(d["X"])
	    print(predictions, d["y"])	

if __name__ == '__main__':
	main()
