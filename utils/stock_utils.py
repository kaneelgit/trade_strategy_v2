"""
Author - Kaneel Senevirathne
Date - 09/16/2022
Description - This script contains all the utility functions used in the stock bot.
"""
from td.client import TDClient
import yahoo_fin.stock_info as si
import pandas as pd
import requests, time, re, os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import os, sys
from datetime import timedelta, datetime
#add parent dir
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import parsers
from scipy.signal import argrelextrema
import tqdm
from parsers import train_vars
args = train_vars()
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

from matplotlib.lines import Line2D

def timestamp(dt):
    """
    Creates a timestamp given an datetime object
    """
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)

def linear_regression(x, y):
    """
    Initiates Linear regression and fit the function given x and y data. Output is the coefficient of the fitted model.
    """
    lr = LinearRegression()
    lr.fit(x, y)

    return lr.coef_[0][0]

def earnings_df(stock):
    """
    Creates the earnings dataframe given the stock.
    """
    earnings_hist = si.get_earnings_history(stock)
    df = pd.DataFrame.from_dict(earnings_hist)
    return df

def get_earnings(df, date):
    """
    This function inputs a stock and a date. Then gets earnings history closest to that day. Returns the stock eps, surprise in the last quarter.
    Also returns the earnings growth in the last four quarters.
    """
    #some functions
    def convert_datetime(x):
        x = datetime.strptime(x[:10], "%Y-%m-%d")
        return x

    def check_closest(x):
        d = (x - date).days
        return d

    #apply changes to dataframe    
    df['date'] = df['startdatetime'].apply(convert_datetime)
    df['closest'] = df['date'].apply(check_closest)
    df = df[['date', 'epsactual', 'epssurprisepct', 'closest']]
    #get last four quarter data
    df_l4 = df[df['closest'] < 0][:4].reset_index(drop = True)

    #get data points epsactual, epssurprise
    eps_actual = df_l4['epsactual'][0]
    eps_surprise = df_l4['epssurprisepct'][0]

    #create data for linear regression
    y = df_l4['epsactual'].to_numpy().reshape(len(df_l4), -1)
    X = np.arange(0, len(df_l4)).reshape(len(df_l4), -1)
    eps_growth = linear_regression(X, y)
    
    return eps_actual, eps_surprise, eps_growth

def get_stock_price(stock, date):
    """
    returns the stock price given a date
    """
    start_date = date - timedelta(days = 7)
    end_date = date
    
    #enter url of database
    url = f'https://api.tdameritrade.com/v1/marketdata/{stock}/pricehistory'

    query = {'apikey': str(parsers.API_KEY), 'startDate': timestamp(start_date), \
            'endDate': timestamp(end_date), 'periodType': 'year', 'frequencyType': \
            'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}

    #request
    results = requests.get(url, params = query)
    data = results.json()
    
    try:
        #change the data from ms to datetime format
        data = pd.DataFrame(data['candles'])
        data['date'] = pd.to_datetime(data['datetime'], unit = 'ms')
        return data['close'].values[-1]
    except:
        print(f'Was not able to get stock price for {stock}. ({datetime.now()})')
        pass

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            
            y = df['close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value
            
    return df

def volume_bs(volume, open, close):

    if (close - open) <= 0:
        v = -1
    else:
        v = 1

    return volume * v

def n_day_regression_v2(df):
    """
    create regressions
    """
    df['lr3'] = df['close'].rolling(3).apply(lambda x: stats.linregress(x.index, x)[0])
    df['lr5'] = df['close'].rolling(5).apply(lambda x: stats.linregress(x.index, x)[0])
    df['lr10'] = df['close'].rolling(10).apply(lambda x: stats.linregress(x.index, x)[0])
    df['lr20'] = df['close'].rolling(20).apply(lambda x: stats.linregress(x.index, x)[0])
    df['volume2'] = df.apply(lambda x: volume_bs(x.volume, x.open, x.close), axis = 1)

    return df

def normalized_values(high, low, close):
    """
    normalize the price between 0 and 1.
    """
    #epsilon to avoid deletion by 0
    epsilon = 10e-10
    
    #subtract the lows
    high = high - low
    close = close - low

    return close/(high + epsilon)

def get_data(stock, start_date = None, end_date = None, n = 10):
    """
    This function gets the required data given the stock, date range. n value is the amount of days. Start date and end date both are required.
    """
    if None in (start_date, end_date):
        print('Need both start and end date')
        return None, None, None

    #enter url
    url = f'https://api.tdameritrade.com/v1/marketdata/{stock}/pricehistory'
    
    payload = {'apikey': str(parsers.API_KEY),'startDate': timestamp(start_date), \
            'endDate': timestamp(end_date), 'periodType': 'year', 'frequencyType': \
            'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}
 
    #request
    results = requests.get(url, params = payload)
    data = results.json()

    #change the data from ms to datetime format
    data = pd.DataFrame(data['candles'])
    data['date'] = pd.to_datetime(data['datetime'], unit = 'ms')

    #add the noramlzied value function and create a new column
    data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis = 1)
    
    #column with local minima and maxima
    data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
    data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order = n)[0]]['close']

    #idx with mins and max
    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    return data, idx_with_mins, idx_with_maxs

def create_dataset(stock, df, start, end):
    """
    Get training and testing datasets.
    """
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, start, end)

    data = n_day_regression_v2(data)

    #get month to the dataset
    def get_month(x):
        return x.month
    data['month'] = data['date'].apply(get_month)
    
    data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop = True) #think about changing the target to 1 here.
    data_['target'] = [1 if x > 0 else 0 for x in data_.loc_max]
    cols_of_interest = ['volume2', 'month', 'date', \
        'normalized_value', 'lr3', 'lr5', 'lr10', 'lr20', 'target']
    data_ = data_[cols_of_interest]
    data_ = data_.dropna(axis = 0).reset_index(drop = True)
   
    #create three empty columns
    data_['eps_actual'] = np.nan
    data_['eps_surprise'] = np.nan
    data_['eps_growth'] = np.nan
   
    #get earnings for each data (find a better way to do this)
    for i in range(len(data_)):
        eps_actual, eps_surprise, eps_growth = get_earnings(df, data_['date'][i])
        data_['eps_actual'][i] = eps_actual
        data_['eps_surprise'][i] = eps_surprise
        data_['eps_growth'][i] = eps_growth
    
    cols_of_interest = ['volume2', 'month', 'date', \
        'normalized_value', 'lr3', 'lr5', 'lr10', 'lr20', 'eps_actual', 'eps_surprise',
       'eps_growth', 'target']

    data_ = data_[cols_of_interest].dropna(axis = 0)   
    
    return data_

def create_train_test_set(stock_list, args):
    """
    This function creates a train test set given the test date range.
    """
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for stock in tqdm.tqdm(stock_list):

        try:
            #get test_date_range
            test_date_range = args.test_date_range
            test_start = datetime.strptime(test_date_range[0], '%m/%d/%Y')
            test_end = datetime.strptime(test_date_range[1], '%m/%d/%Y')
            
            #data range
            data_start = datetime(2008, 1, 1)
            data_end = datetime(2022, 9, 1)

            #get training data range
            train_start_1 = data_start
            train_end_1 = test_start - timedelta(days = 1)
            train_start_2 = test_end + timedelta(days = 1)
            train_end_2 = data_end
            
            #get earnings df for the stock
            df = earnings_df(stock)

            #create train and test data
            train_data1 = create_dataset(stock, df, train_start_1, train_end_1)
            train_data2 = create_dataset(stock, df, train_start_2, train_end_2)
            train_data = pd.concat([train_data1, train_data2], axis = 0).reset_index(drop = True)

            test_data = create_dataset(stock, df, test_start, test_end)

            train_df = pd.concat([train_df, train_data], axis = 0).reset_index(drop = True)
            test_df = pd.concat([test_df, test_data], axis = 0).reset_index(drop = True)
            
            #sleep for 2 seconds 
            time.sleep(2)
        except:
            print(f'{stock} data could not be fetched')
            time.sleep(5)

    return train_df, test_df

def get_data_for_date(stock, end_date, earnings_dataframe):
    """
    Prepares a the dataset to input to the model given the date
    """
    #get data
    start_date = end_date - timedelta(days = 45)
    data, _, _ = get_data(stock, start_date, end_date)
    idxs = np.arange(0, len(data))

    #create regressions for 3, 5, 10 and 20 days
    data = n_day_regression_v2(data)
    
    #get month to the dataset
    def get_month(x):
        return x.month
    data['month'] = data['date'].apply(get_month)

    #get only the last index (day)
    data = data[-1:].reset_index(drop = True)
    
    #get earnings dataframe
    eps_actual, eps_surprise, eps_growth = get_earnings(earnings_dataframe, data['date'][0])
    data['eps_actual'] = eps_actual
    data['eps_surprise'] = eps_surprise
    data['eps_growth'] = eps_growth
    
    #columns needed in order    
    cols_of_interest = ['volume2', 'month', \
        'normalized_value', 'lr3', 'lr5', 'lr10', 'lr20', 'eps_actual', 'eps_surprise',
       'eps_growth']
    
    return data[cols_of_interest], data['close'][0]

def _threshold(probs, threshold):
    """
    Inputs the probability and returns 1 or 0 based on the threshold
    """
    prob_thresholded = [0 if x > threshold else 1 for x in probs[:, 0]]

    return np.array(prob_thresholded)

def create_plot(stock, scaler, model, args, save_dir = None):
    """
    Creates a plot with predictions given the scaler and the model
    """
    try:

        #get test_date_range
        test_date_range = args.test_date_range
        test_start = datetime.strptime(test_date_range[0], '%m/%d/%Y') 
        start = test_start - timedelta(days = 100)
        test_end = datetime.strptime(test_date_range[1], '%m/%d/%Y') 
        end = test_end + timedelta(days = 100)

        #predictions
        predicted_days = []
        predicted_day_price = []

        #get data for the stock
        start_date = test_start
        delta = timedelta(days = 1)

        #get earnings dataframe
        earnings_dataframe = earnings_df(stock)

        while start_date <= test_end:
            
            #get data for the date
            data, close_price = get_data_for_date(stock, start_date, earnings_dataframe)

            #scale the data
            input_data_scaled = scaler.transform(data.to_numpy())
            prediction = model._predict_proba_lr(input_data_scaled)
            prediction_thresholded = _threshold(prediction, args.threshold)

            #if pred thresh is 0 buy signal
            if prediction_thresholded[0] < 1:
                predicted_days.append(start_date)
                predicted_day_price.append(close_price)

            start_date += delta
    
        #get data to plot
        data, _, _ = get_data(stock, start, end)
        
        #plot the figure
        plt.figure()
        plt.plot(data['date'], data['close'], color = 'k', linewidth = 0.5)
        plt.scatter(predicted_days, predicted_day_price, color = 'green', s = 50, alpha = 1)
        plt.xticks(rotation = 90)
        plt.fill_between([start, test_start], [data['close'].min()], [data['close'].max()], color = 'grey', alpha = 0.7)
        plt.fill_between([test_end, end], [data['close'].min()], [data['close'].max()], color = 'grey', alpha = 0.7)
        plt.fill_between([test_start, test_end], [data['close'].min()], [data['close'].max()], color = 'grey', alpha = 0.1)
        plt.title(f'{stock} predictions')
        dark_grey = mpatches.Patch(color = 'grey', alpha = 0.7, label = 'Train data')
        light_grey = mpatches.Patch(color = 'grey', alpha = 0.2, label = 'Test data')
        red_circle = Line2D([0], [0], marker='o', color='w', label='Predicted buying opportunities',
                            markerfacecolor='green', markersize=15)
        plt.legend(handles = [dark_grey, light_grey, red_circle])
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir)
        else:
            plt.savefig('test.png')
    
    except:
        pass
