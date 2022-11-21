"""
Author - Kaneel Senevirathne
Date - 11/20/2022
"""
import argparse

#get credentials file
with open('/mnt/c/Users/kanee/Desktop/git/trading_strategy_v2/credentials.txt', 'r') as f:
    API_KEY = f.readlines()[0]

dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',\
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',\
        'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']

def train_vars():
    """
    Input variables for model training
    """
    parser = argparse.ArgumentParser(description = "Required/optional arguments for model training")
    parser.add_argument('--test_date_range', nargs = "+", help = 'This is the date range to create the test set. Expected input is a list containing two dates\
        in a list. Format is "m/d/y". Ex: ["01/01/2021", "12/01/2021"]. The train will be created based on the given test set range', \
            default = ["01/01/2021", "12/01/2021"])
    parser.add_argument('--model_name', help = 'Name the model. (i.e "v1")', default = 'v1', type = str)
    parser.add_argument('--threshold', help = 'Threshold value for the trained model. Default value is 0.98', default = 0.98, type = float)
    args = parser.parse_args()
    return args