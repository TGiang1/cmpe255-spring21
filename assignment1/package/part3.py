import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from package import utility as ut


# This function trains and tests a multiple regression model on the data with multiple features. Specify the features by passing in array of column names (string)
def multipleRegression(data, col_names):
    #Handle splitting the data

    X = data[col_names]
    y = data['MEDV'] #median value of house is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

    #Training the multiple regression model
    model = LinearRegression() #We still use a linear regression model, but there are now multiple feature columns (multiple regression)
    model.fit(X_train, y_train)

    #Calculate RMSE Score
    y_pred = model.predict(X_test)
    print("Using multiple regression the reported RMSE between the predicted MEDV results and the actual results is", mean_squared_error(y_test, y_pred, squared=False))
    
    #Calculate R2 Score
    print("And, the reported R-squared score between the predicted MEDV and actual MEDV is", r2_score(y_test, y_pred))
    

    #Calculate adjust R2 Score
    r2 = r2_score(y_test, y_pred)
    p = float(6) #hard code using 6 features
    N = float(405) #hard code sample size is 405.
    Adjusted_r2 = 1 - (( 1- r2) * (N-1) / (N - p - 1))
    print("And the adjusted R-squared score is ", Adjusted_r2)

# This function does everything required of part3. We test Multiple Regression with INDUS, NOX, RM, TAX, PTRATIO, and LSTAT
def run(data):
    print("Part3:\nBuilding and Testing a Multiple Regression model with 6 features (INDUS, NOX, RM, TAX, PTRATIO, LSTAT)...")
    p3_r2_score = multipleRegression(data, ['INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']) 

    print("\n")