

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion

import os
import json

with open('config.json','r') as f:
    config = json.load(f) 


prod_deployment_path = os.path.join(config['prod_deployment_path']) 
input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = config['output_folder_path']

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
proceed = 0
if len(new_files) > 0:
    proceed = 1
    ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

if __name__ == "__main__":

    with open('config.json','r') as f:
        config = json.load(f) 
        ##################Check and read new data
    #first, read ingestedfiles.txt
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 
    input_folder_path = os.path.join(config['input_folder_path'])
    output_folder_path = config['output_folder_path']

    list_files = list()
    with open(prod_deployment_path + "/" + "ingestedfiles.txt", "r") as ingested_df:
        list_files = ingested_df.readlines()

    list_files = [x.rstrip("\n") for x in list_files] # remove trailing symbol

    for (_, _, filenames) in os.walk(input_folder_path):
        meta_data_list = [filename for filename in filenames if filename.endswith(".csv")]
    
    new_files = [filename for filename in meta_data_list if filename not in list_files]
    
    if len(new_files) > 0:
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)



