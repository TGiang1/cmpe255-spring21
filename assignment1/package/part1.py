
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from package import utility as ut

# This function plots the data (X ,Y) along with the best fit line (obtained from the model)
def plotLine(X, Y, model, feature_name):
    plt.figure(figsize = (12, 8)) #Set figure size
    plt.scatter(X, Y, color='b') #Plot the data points

    #Set labels
    plt.xlabel(feature_name)
    plt.ylabel('Median_house_value')
    plt.title('Median_house_value in $1000s vs.' + feature_name)

    #Plot best fit line.
    minimum = X.min() #Get minimum and maximum of feature data
    maximum = X.max()
    input =  np.linspace(minimum,maximum,100) #Create an input array of 100 elements between minimum and maximum of data point input
    plt.plot(input, model.predict(input), color='k')

    plt.show(block=True)


# This function trains and tests a LinearRegression model on the data with using only 1 feature. Specify the feature by passing in feature name to col_name
# Returns r2_score because we will need it for adjusted r2_score in part3
def linearRegression(data, col_name):
    #Handle splitting the data

    y = data['MEDV'] #median value of house is the target variable
    X = data[[col_name]] # col_name is the chosen feature
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

    #Training the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Plotting the data (train and test) against the best fit line
    print("Let us plot the training data with the best fit line...")
    plotLine(X_train, y_train, model, col_name)

    print("Now, let us plot the test data with the best fit line...")
    plotLine(X_test, y_test, model, col_name)

    #Calculate RMSE Score
    y_pred = model.predict(X_test)
    print("Using", col_name, "as the feature and linear regression model, the reported RMSE between the predicted MEDV results and the actual results is", mean_squared_error(y_test, y_pred, squared=False))
    
    #Calculate R2 Score
    print("And, the reported R-squared score between the predicted MEDV and actual MEDV is", r2_score(y_test, y_pred))
    #print(y_train.name)
    #print(X_train[col_name].name)
    #y_pred = regressor.predict(X_test)
    return r2_score(y_test, y_pred)

# This function is an add on to part1. We loop through every feature of the dataframe.
# For each loop, we train the linear regression model, plot the data (train and test) against this model, and calculate the RMSE score and R2 score.
def run_loop(data):
    print("Extra:")
    print("Let us loop through all of the features and try Linear Regression model on it to see how well it performs for predicting MEDV!")
    columns = data.columns
    for col in columns:
        linearRegression(data, col)

    print("As we can see from looping through each feature and using each as the single feature for training the Linear Regression model, the RM feature did the best!")
    print("When using RM, the Linear Regression model performed with a RMSE score and r-squared score of 6.38 and 0.5877 respectively.")

# This function does everything required of part1. We test LinearRegression with RM.
# Returns r2_score because we will need it for adjusted r2_score in part2
def run(data):
    print("Part1:\nBuilding and Testing a Linear Regression model with 1 feature...")
    ut.viewCorrHeatmap(data)
    print("Since this feature has a high correlation score to MEDV, let us choose RM as the feature to train and test the Linear Regression model.")
    print("Train LinearRegression model using just RM, or average number of rooms per dwelling.")
    r2_score = linearRegression(data, 'RM') 

    print("\n")
    return r2_score
