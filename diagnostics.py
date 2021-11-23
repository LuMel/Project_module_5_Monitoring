
import pandas as pd
import numpy as np
import subprocess
import timeit
from typing import List, Tuple, Optional
from io import BytesIO
import os
import json
import pickle
from training import train_model
from ingestion import merge_multiple_dataframe

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deployment_path = os.path.join(config['prod_deployment_path']) 


##################Function to get model predictions
def model_predictions(data: Optional[pd.DataFrame] = None) -> np.ndarray:
    # read the deployed model and a test dataset, calculate predictions
    with open(deployment_path + "/" + "trainedmodel.pkl", 'rb') as model_pickle:
        model_ = pickle.load(model_pickle)
    # read scaler
    with open(deployment_path + "/" + "scscaler.pkl", 'rb') as pickle_sc:
        sc_ = pickle.load(pickle_sc)
    
    if data is None:
        data = pd.read_csv(test_data_path + "/" + "testdata.csv")
    X_test = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    return model_.predict(sc_.transform(X_test))

##################Function to get summary statistics
def dataframe_summary() -> List[Tuple[float, float, float]]:
    
    data = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    # calculate summary statistics here
    # https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas
    data_numeric = data.select_dtypes(include=np.number)
    list_stats = list()
    
    for col in data_numeric:
        list_stats.append((data_numeric[col].mean(), data_numeric[col].median(), data_numeric[col].std()))
    
    return list_stats

##################Function to check for missing data
def missing_data():

    data = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    return ((pd.isnull(data).sum())/len(data)).values


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_tr = timeit.default_timer()
    train_model()
    end_tr = timeit.default_timer()

    start_ing = timeit.default_timer()
    merge_multiple_dataframe()
    end_ing = timeit.default_timer()
    return [end_tr - start_tr, end_ing - start_ing]

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list','--outdated'])
    df_ = pd.read_csv(BytesIO(outdated_packages), sep = "\t")
    
    return df_


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()

    
