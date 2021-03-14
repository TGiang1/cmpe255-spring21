
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from package import utility as ut

# This function plots the data (X ,Y) along with the best fit curve (obtained from the model)
def plotCurve(X, Y, model, feature_name, degree):
    plt.figure(figsize = (12, 8)) #Set figure size
    plt.scatter(X.iloc[:,1], Y, color='b') #Plot the data points

    #Set labels
    plt.xlabel(feature_name)
    plt.ylabel('Median_house_value')
    plt.title('Median_house_value in $1000s vs.' + feature_name)

    #Plot best fit curve.
    minimum = X.iloc[:,1].min() #Get minimum and maximum of feature data
    maximum = X.iloc[:,1].max()
    input =  np.linspace(minimum,maximum,100) #Create an input array of 100 elements between minimum and maximum of data point input
    input = input.reshape(100,1) #reshape the input array in order to poly transform it

    poly = PolynomialFeatures(degree)
    input = poly.fit_transform(input) #Array of three columns. First column is input^0, second column is input^1, third column is input^2.

    plt.plot(input[:,1], model.predict(input), color='k')

    plt.show(block=True)


# This function trains and tests a PolyRegression model on the data with using only 1 feature. Specify the feature by passing in feature name to col_name
# This function uses PolynomialFeatures to generate the polynomial features of the chosen column, (which will be RM)
# The degree is 2 (quadratic)
def polyRegression(data, col_name, degree):
    #Handle splitting the data

    y = data['MEDV'] #median value of house is the target variable
    X = data[[col_name]] # col_name is the chosen feature

    #Generate new feature matrix consisting of all polynomial combinations of the chosen X feature.
    poly = PolynomialFeatures(degree)
    X_new = poly.fit_transform(X) #X_new is a numpy array, now with RM feature along with two additional columns (to the power of 0, and to the power of 2)
    X = pd.DataFrame(data=X_new) #Transform back into dataframe.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

    #Training the polynomial regression model
    model = LinearRegression() #We still use a linear regression model, but the features now include RM^0, RM^1, RM^2, which means we are essentially fitting a quadratic curve.
    model.fit(X_train, y_train)

    #Plotting the data (train and test) against the best fit line
    print("Let us plot the training data with the best fit curve...")
    plotCurve(X_train, y_train, model, col_name, degree)

    print("Now, let us plot the test data with the best fit curve...")
    plotCurve(X_test, y_test, model, col_name, degree)

    #Calculate RMSE Score
    y_pred = model.predict(X_test)
    print("Using", col_name, "as the feature and polynomial regression model, the reported RMSE between the predicted MEDV results and the actual results is", mean_squared_error(y_test, y_pred, squared=False))
    
    #Calculate R2 Score
    print("And, the reported R-squared score between the predicted MEDV and actual MEDV is", r2_score(y_test, y_pred))

# This function is an add on to part2. We loop through every feature of the dataframe.
# For each loop, we train the poly regression model, plot the data (train and test) against this model, and calculate the RMSE score and R2 score.
def run_loop(data):
    print("Extra:")
    print("Let us loop through all of the features and try Poly Regression model (deg 2) on it to see how well it performs for predicting MEDV!")
    columns = data.columns
    for col in columns:
        polyRegression(data, col, 2)

    print("As we can see from looping through each feature and using each as the single feature for training the Poly Regression model, the RM feature still did the best!")
    print("When using RM as the feature, the Poly Regression model performed with a RMSE score and r-squared score of 5.799 and 0.6596 respectively.")

# This function does everything required of part2. We test PolynomialRegression with RM.
def run(data):
    print("Part2:\nBuilding and Testing a Polynomial Regression model with 1 feature...")
    print("We will use the same feature as in part1. We will test with RM for training the Polynomial Regression model.")
    print("Train PolynomialRegression model using just RM, or average number of rooms per dwelling.")
    polyRegression(data, 'RM', 2) #Degree is two, quadratic curve for model.

    print("\nNow, let us plot PolynomialRegression model with degree 20.")
    polyRegression(data, 'RM', 20) 

    print("\n")
    