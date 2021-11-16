from flask import Flask, session, jsonify, request
import pandas as pd
from pandas.io.pickle import to_pickle
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():

    clean_data = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    sc = StandardScaler()

    X = clean_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = clean_data["exited"].values
    X_sc = sc.fit_transform(X)
    #use this logistic regression for training
    logit_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    logit_model.fit(X_sc, clean_data["exited"])
    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path + "/" + "trainedmodel.pkl", 'wb') as pickle_:
        pickle.dump(logit_model, pickle_, protocol=pickle.HIGHEST_PROTOCOL)
    # write scaler
    with open(model_path + "/" + "scscaler.pkl", 'wb') as pickle_sc:
        pickle.dump(sc, pickle_sc, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    train_model()
    
