import numpy as np
import os
import pandas as pd
import time

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "FIXME"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

def random_digit(X):
    test_array = X.to_numpy()
    np.random.seed( int(time.time())) #set the random seed based on the time
    
    random_entry = np.random.randint(0, np.shape(test_array)[0], size=1) #get random index number within boundary of test set passed in
    random_entry = random_entry[0] #get the random index number
    print("This random digit is from entry", random_entry, "of the dataframe testset passed in.")

    some_digit = test_array[random_entry] #Get the entry of the random number generated
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary,
            interpolation="nearest")
    plt.axis("off")

    save_fig("some_digit_plot") #save the random entry digit
    plt.show()
    return [some_digit, random_entry] #return the random digit data as well as its index

#we will not used this because it is fetching the mnist dataset from sklearn. Instead, we use a local copy of the mnist dataset.
def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
        sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    mnist["data"], mnist["target"]


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def train_model(X_train, y_train):
    # Example: Binary number 4 Classifier
    y_train_4 = (y_train == 4) #set the target output to be 0 or 1 depending on if it is a 4 or not.

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    clf.fit(X_train, y_train_4)

    return clf

def predict(some_digit, model):
    some_digit = some_digit.reshape(1, -1) #reshape since single sample
    prediction = model.predict(some_digit)

    if (prediction[0] == True):
        print("The model has predicted that this digit is a 4!")
    else:
        print("The model has predicted that this digit is not a 4.")
    
#This function calculates the cross validation score of the model on the entire training dataset
def calculate_cross_val_score(model, X, y):
    y_is_4 = (y == 4)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    scores = cross_val_score(clf, X, y_is_4, cv=3)
    print("The scores of 3 different cross_validation splits are:")
    print(scores)
    print("The mean score and standard deviation is:")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


def test():
    print("Loading the mnist_train.csv dataset...\n")
    data = pd.read_csv('mnist_train.csv') #Get training dataset
    print("Splitting the dataset into X (features dataframe) and y (target label)...\n")
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    print("The first 5 entries of the features dataframe (pixel values) is:")
    print(X.head().to_markdown())

    print("\nThe first 5 entries target label series looks like:")
    print(y.head().to_markdown())

    print("Splitting the dataset into training and test set...\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #split the dataset into train and test set

    print("Training the binary classification model (whether or not number is a 4) using SDClassifier on test set...\n")
    model = train_model(X_train, y_train)

    print("Selecting a random entry from the test set...\n")
    some_digit_and_index = random_digit(X_test)
    some_digit = some_digit_and_index[0]
    index = some_digit_and_index[1]

    print("\nPredicting whether this random digit is a 4 or not using our SDClassifier model...")
    predict(some_digit, model)

    y_test = y_test.reset_index(drop=True) #reset index of y_test so we can get the correct entry based on index
    actual_number = y_test[index]
    print("The actual number was a", actual_number)

    print("\nLet us test the cross validation score of our model (4 classifier) on the entire dataset...")
    calculate_cross_val_score(model, X, y)

if __name__ == "__main__":
    # execute only if run as a script
    test()
