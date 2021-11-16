
import pandas as pd
import numpy as np
import timeit
from typing import List, Tuple
import os
import json
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deployment_path = os.path.join(config['prod_deployment_path']) 


##################Function to get model predictions
def model_predictions(data: pd.DataFrame) -> np.ndarray:
    # read the deployed model and a test dataset, calculate predictions
    with open(deployment_path + "/" + "trainedmodel.pkl", 'rb') as model_pickle:
        model_ = pickle.load(model_pickle)
    # read scaler
    with open(deployment_path + "/" + "scscaler.pkl", 'rb') as pickle_sc:
        sc_ = pickle.load(pickle_sc)

    return model_.predict(sc_.transform(data))

##################Function to get summary statistics
def dataframe_summary(data: pd.DataFrame) -> List[Tuple[float, float, float]]:
    #calculate summary statistics here
    # https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
    data_numeric = data.select_dtypes(include=np.number)
    list_stats = list()

    for col in data_numeric:
        list_stats.append((data_numeric[col].mean, data_numeric[col].median, data_numeric[col].std))

    return list_stats

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
