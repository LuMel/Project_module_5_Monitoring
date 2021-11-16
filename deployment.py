from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import shutil
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle():#model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copyfile(model_path + "/" + "trainedmodel.pkl",  
                    prod_deployment_path + "/" + "trainedmodel.pkl")
    shutil.copyfile(model_path + "/" + "scscaler.pkl",  
                    prod_deployment_path + "/" + "scscaler.pkl")
    shutil.copyfile(dataset_csv_path + "/" + "ingestedfiles.txt",  
                    prod_deployment_path + "/" + "ingestedfiles.txt")
    shutil.copyfile(model_path + "/" + "latestscore.txt",  
                    prod_deployment_path + "/" + "latestscore.txt")
    

if __name__ == "__main__":
    store_model_into_pickle()
        
        

