from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary
from diagnostics import missing_data, execution_time, outdated_packages_list
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction")#, methods=['POST','OPTIONS'])
def predict():
    data = pd.read_csv(test_data_path + "/" + "testdata.csv")
    #call the prediction function you created in Step 3
    return str(model_predictions(data)) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    return str(score_model(False))

#######################Summary Statistics Endpoint
@app.route("/summarystats")#, methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return dataframe_summary()

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    return missing_data(), execution_time(), outdated_packages_list()

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
