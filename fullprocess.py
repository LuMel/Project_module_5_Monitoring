

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import pandas as pd
import app
import time
import apicalls

import os
import json

with open('config.json','r') as f:
    config = json.load(f) 


prod_deployment_path = os.path.join(config['prod_deployment_path']) 
input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = config['output_folder_path']


def run_process():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    list_files = list()
    with open(prod_deployment_path + "/" + "ingestedfiles.txt", "r") as ingested_df:
        list_files = ingested_df.readlines()

    list_files = [x.rstrip("\n") for x in list_files] # remove trailing symbol

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    for (_, _, filenames) in os.walk(input_folder_path):
        meta_data_list = [filename for filename in filenames if filename.endswith(".csv")]

    new_files = [filename for filename in meta_data_list if filename not in list_files]

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    proceed_modeldrift = False
    if len(new_files) > 0:
        proceed_modeldrift = True
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)


    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    proceed_redeployment = False

    if proceed_modeldrift:
        with open(prod_deployment_path + "/" + "latestscore.txt", "r") as latestscore_df:
            score = float(latestscore_df.read())

        # load recently-ingested data
        updated_data = pd.read_csv(output_folder_path + "/finaldata.csv")
        # compute f1 score
        new_f1score = scoring.score_model(save=False, test_data=updated_data, path=prod_deployment_path)


        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        if new_f1score < score:
            proceed_redeployment = True


    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    if proceed_redeployment:
        # train_model will automatically pick the newest data from the correct folder
        # and save it.
        training.train_model()
        scoring.score_model(save=True)
        #redeploy
        deployment.store_model_into_pickle()


        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
        reporting.score_model()

        # the app must be up and running for this to work!
        apicalls.write_returns()

if __name__ == "__main__":
    run_process()