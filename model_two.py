"""
Author - Kaneel Senevirathne
Date - 11/20/2022
"""
import os, sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import parsers
from utils.stock_utils import create_train_test_set, create_plot, create_plot_v2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd
from tqdm import tqdm


class LR_training:

    def __init__(self, args):
        
        #initialize vars
        self.args = args
        self.model_name = args.model_name
        self.threshold = args.threshold
        self.stock_list = parsers.dow
        # self.stock_list = ["BA", "AAPL"]
        #initialize models
        self.scaler = MinMaxScaler()
        self.lr = LogisticRegression()

        #run training functions
        self.fetch_train_test_data()
        self.fit_model()
        self.create_confusion_matrix()
        self.save_model()
        self.save_results()

    def fetch_train_test_data(self):
        """
        This function creates train and test data
        """
        print('Creating train and test sets..\n')
        #get train and test dataframes
        train_df, test_df = create_train_test_set(self.stock_list, self.args)
        self.columns = test_df.columns

        #reshuffle the train dataset
        train_df = train_df.sample(frac = 1, random_state = 3).reset_index(drop = True)
        train_Y = train_df.pop('target').to_numpy()
        self.train_Y = train_Y.reshape(train_Y.shape[0], 1)
        self.train_X = self.scaler.fit_transform(train_df)

        #test set
        test_Y = test_df.pop('target').to_numpy()
        self.test_Y = test_Y.reshape(test_Y.shape[0], 1)
        self.test_X = self.scaler.transform(test_df)
        
        print(f'Train set shape: {train_Y.shape[0]}')
        print(f'Test set shape: {test_Y.shape[0]}\n')

    def fit_model(self):
        """
        Use train data to fit the LR model
        """
        print('Training model..\n')
        self.lr.fit(self.train_X, self.train_Y)

        #get predictions for the test data and find the score
        self.predictions = self.lr.predict(self.test_X)
        self.score = self.lr.score(self.test_X, self.test_Y)
        print(f'Logistic regression model score: {self.score}\n')

        #preds with threshold
        self.predictions_proba = self.lr._predict_proba_lr(self.test_X)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)

    def _threshold(self, predictions, threshold):
        """
        Use a different threshold to find the predictions
        """
        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]
        return np.array(prob_thresholded)

    def create_confusion_matrix(self):
        """
        Create a confusion matrix given test labels and predictions
        """
        cm = confusion_matrix(self.test_Y, self.predictions)
        self.cmd = ConfusionMatrixDisplay(cm)
        
        cm_thresholded = confusion_matrix(self.test_Y, self.predictions_proba_thresholded)
        self.cmd_thresholded = ConfusionMatrixDisplay(cm_thresholded)

    def save_model(self):
        """
        save the model in the saved models folder.
        """
        saved_models_dir = 'saved_models'
        model_file = f'lr_{self.model_name}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_name}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {saved_models_dir}\n')

    def save_results(self):
        """
        save results in the results folder
        """
        folder_name = f'lr_{self.model_name}'
        results_dir = os.path.join('results', f'{folder_name}')
      
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        pred_results_dir = os.path.join(results_dir, 'pred_results')
        
        if not os.path.exists(pred_results_dir):
            os.makedirs(pred_results_dir)

        #save confusion matrices
        plt.figure()
        self.cmd.plot()
        plt.savefig(f'{results_dir}/cm_{self.model_name}.jpg')

        plt.figure()
        self.cmd_thresholded.plot()
        plt.savefig(f'{results_dir}/cm_thresholded_{self.model_name}.jpg')

        #create summary dict
        summary_dict = {
            'score': self.score,
            'test_date_range': self.args.test_date_range,
            'threshold': self.threshold
        }

        #save in a text file
        file = open(f'{results_dir}/results.txt', 'w')
        for key, value in summary_dict.items():
            file.write('%s:%s\n' % (key, value))
        
        file.close()
        print(f'Figures and summary saved in {results_dir}')

        #save coefficients
        predictor_names = self.columns[:len(self.columns)-1]
        coefficients = self.lr.coef_.ravel()
        coef = pd.Series(coefficients, predictor_names).sort_values()
        plt.figure()
        coef.plot(kind = 'bar', title = 'Coefficients')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/coefficients.jpg')

        #create plot
        for stock in tqdm(self.stock_list):
            save_dir = os.path.join(pred_results_dir, str(stock))
            create_plot_v2(stock, self.scaler, self.lr, self.args, save_dir)


if __name__ == "__main__":

    #import args
    args = parsers.train_vars()
    lr_training = LR_training(args)

    
