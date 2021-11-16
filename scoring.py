from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    #load fitted model
    with open(model_path + "/" + "trainedmodel.pkl", "rb") as model_file:
        model_ = pickle.load(model_file)
    # load StandardScaler
    with open(model_path + "/" + "scscaler.pkl", "rb") as sc_file:
        sc_ = pickle.load(sc_file)

    test_data = pd.read_csv(test_data_path + "/" + "testdata.csv")
    X_test = test_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = test_data["exited"].values
    Xtest_sc = sc_.transform(X_test)

    preds = model_.predict(Xtest_sc)

    with open(model_path + "/" + "latestscore.txt", 'w') as outfile:
        outfile.write(str(metrics.f1_score(y_test, preds)))
    
if __name__ == "__main__":
    score_model()

