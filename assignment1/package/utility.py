
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

#This file contains two methods that other modules can use.
#load_data reads the boston_housing data into a dataframe object and returns it
#viewCorrHeatmap shows a correlation heatmap of the inputted dataframe

def load_data():
    print("Loading the data into dataframe format...")
    data = pd.read_csv('housing.csv', header=None, sep=r'\s{1,}', engine='python') #Read Boston Housing Dataset. Columns are separated by 1 or more whitespaces
    data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] #Set the headers for the dataframe columns
    print("The first 5 entries are:\n", data.head().to_markdown(), "\n")
    print("The data info is:\n")
    data.info()
    print("\nThe data has been loaded.\n")
    return data

def viewCorrHeatmap(data):
    print("Taking a look at the correlation heatmap...")
    corrs = data.corr()
    plt.figure(figsize = (14, 10))
    sns.heatmap(corrs, annot = True, vmin = -1, vmax = 1, fmt = '.3f', cmap=plt.cm.PiYG_r)
    plt.show(block=True)