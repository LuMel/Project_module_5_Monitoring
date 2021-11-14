import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe(input_path_: str = input_folder_path,
                             output_path_: str = output_folder_path):
    #check for datasets, compile them together, and write to an output file
    data_list = list()
    for (dirpath, _, filenames) in os.walk(input_path_):
        data_list = [pd.read_csv(dirpath + "/" + filename) 
                     for filename in filenames if filename.endswith(".csv")]
        meta_data_list = [filename for filename in filenames if filename.endswith(".csv")]

    with open(output_path_ + "/" + 'ingestedfiles.txt', 'w') as f:
        for item in meta_data_list:
            f.write("%s\n" % item)

    final_df = pd.concat(data_list).drop_duplicates()                 
    final_df.to_csv(output_path_ + "/" + "finaldata.csv", index=False)

    return None


if __name__ == '__main__':
    merge_multiple_dataframe()
