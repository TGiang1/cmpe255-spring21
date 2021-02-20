import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import math

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    # This function will check the rmse value between the inputted observed and predicted sets
    def validate(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
        
    # Get the linear regression parameters based on the training set
    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X]) #Apply a column of 1s to get bias term

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y) #Formula to get the array of weight parameters
    
        return w[0], w[1:] #First term is the bias term, and afterwards is the weights for each feature
        

    # This function will prepare our df by returning a new df (in numpy array form) with only the desired features to analyze upon.
    def prepare_X(self, input_data, base): #base should be an array of strings, where each string is a name of a column that should be in the desired list
        df_num = input_data[base]                   #Input_data is the input data set you would like to prepare (train, validation, or test)
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

def test() -> None:
    carPrice = CarPrice()
    carPrice.trim() #Trim the carPrice df to make all column names and data values (if object) into lowercase and replace all spaces with _
    df = carPrice.df

    np.random.seed(2) # Set a random seed
    n = len(df) # n is the number of entries in the df
    n_val = int(0.2 * n) # n_val is the number of entries in validation data set (20% of total entries)
    n_test = int(0.2 * n) # n_test is the number of entries in validation data set (20% of total entries)
    n_train = n - (n_val + n_test) # n_train is the number of entries in validation data set (60% of total entries)

    idx = np.arange(n) # Arrange an array with same length as df entries, and indexes being the range of the array (in order)
    np.random.shuffle(idx) # Shuffle this array 

    df_shuffled = df.iloc[idx] # this df_shuffled will have the entries of original df randomized now
    
    #Split the shuffled df into training, validation, and test sets in ratio of 60:20:20 respectively
    df_train = df_shuffled.iloc[:n_train].copy() 
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()

    #Get the target variable (msrp) arrays for each set
    y_train_orig = df_train.msrp.values
    y_val_orig = df_val.msrp.values
    y_test_orig = df_test.msrp.values

    #Like in the demo, we will
    #Natural log these arrays for a better distribution (new value = ln(old value + 1))
    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)

    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity'] #Like in the demo, we will select only these 5 features to be used in the linear regression model to predict MSRP
    X_train = carPrice.prepare_X(df_train, base) #Prepare our training set so it only has the "base" features above and get it back in form of numpy array

    #Apply linear regression function with our training to get weight parameters for linear regression line.
    w_0, w = carPrice.linear_regression(X_train, y_train)

    #Use our linear regression function to predict MSRP values on our validation set (since it was not used in training our model)
    X_val = carPrice.prepare_X(df_val, base)
    y_pred_val = w_0 + X_val.dot(w)
    print("The rmse value of predicted MSRP and actual MSRP of validation set is ", carPrice.validate(y_val, y_pred_val))

    #Use our linear regression function to predict MSRP values on our test set
    X_test = carPrice.prepare_X(df_test, base)
    y_pred_test = w_0 + X_test.dot(w)
    print("The rmse value of predicted MSRP and actual MSRP of test set is ", carPrice.validate(y_test, y_pred_test))

    #Now, let us print the desired output for lab2 for the Validation Set

    #Convert the predicted MSRP of validation and test set back to their original values (reverse the ln function)
    y_pred_MSRP_val = np.expm1(y_pred_val) # expm1 calculates exp(x) - 1
    
    df_val['msrp_pred'] = y_pred_MSRP_val # Add the column
    
    print("Let us print out first 5 cars in our Validation Set's original msrp vs. predicted msrp")
    print(df_val.iloc[:,5:].head().to_markdown(), "\n")


if __name__ == "__main__":
    # execute only if run as a script
    test()
