import pickle
from sklearn.model_selection import train_test_split
from diagnostics import model_predictions
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    # read the deployed model and a test dataset, calculate predictions
    data_test = pd.read_csv(test_data_path + "/" + "testdata.csv")
    y_real = y_test = data_test["exited"].values
    preds = model_predictions(data_test)

    # one could use metrics.plot_confusion_matrix at this point.
    # However, the spirit of the exercise is differente

    # from https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    df_cm = pd.DataFrame(metrics.confusion_matrix(y_real, preds), 
                 index = ["False", "True"],
                  columns = ["False", "True"])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(model_path + "/" + "confusionmatrix.png")

if __name__ == '__main__':
    score_model()
