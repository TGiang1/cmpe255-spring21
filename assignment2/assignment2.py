import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import confusion_matrix

from sklearn.datasets import fetch_lfw_people

class PeopleImageClassifier:
    def __init__(self) -> None:
        return 
        
    def load_data(self):
        print("Loading the Labeled Faces in the Wild (LFW) dataset...")
        faces = fetch_lfw_people(min_faces_per_person=60)
        print('Data loaded.')
        print(f"Data set target names is {faces.target_names}\n")
        self.people = faces.target_names #Store the people list as a class attribute
        self.X = faces.data #Store image data in this class object's X variable
        self.y = faces.target #Store target labels in this class object's y variable

    def explore_data(self):
        print("Exploring the loaded data...")
        print(f"The shape of the image data is {self.X.shape}")
        print(f"The minimum value of the image data is {self.X.min()} and the maximum value is {self.X.max()}")
        print(f"The datatype of the image data is {self.X.dtype}")
        print(f"The shape of the label data is {self.y.shape}")
        print(f"The minimum value of the label data is {self.y.min()} and the maximum value is {self.y.max()}\n")

    def print_24_data_images(self):
        print("Printing 24 random images of full dataset...\n")
        plt.figure(figsize=(15,15))

        for i in range(24): 
            ax = plt.subplot(4, 6, i + 1)
            random_int = random.randint(0, self.X.shape[0] - 1) #Get a random integer within bounds of number of X samples. (-1 because randint inclusive)
            chosen_image = self.X[random_int]
            reshaped = chosen_image.reshape((62,47)) #Reshape to image dimensions
            plt.imshow(reshaped)
            label = self.y[random_int]
            person = self.people[label]
            plt.title(person)
            plt.axis("off")
        
        plt.show(block=True)
    
    def split(self):
        print("Splitting into train and test data...")
        #Split data into training and testing data with 80:20 ratio
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        print(f"The shape of the X_train image data is {X_train.shape}")
        print(f"The shape of the X_test image data is {X_test.shape}\n")
        return X_train, X_test, y_train, y_test

    def simple_normalize(self, X_train, X_test):
        print("Normalizing X_train and X_test to make each input vector to be [0,1]...\n")
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        return X_train, X_test

    def make_pipeline_model(self):
        print("Making and returning the single pipeline model with RandomizedPCA and rbf SVM model...") 
        print("We will use RandomizedPCA to extract 150 fundamental components (from the near 3000 pixel features) as step 1 of the pipeline.")
        print("As step 2 of our pipeline we will have a rbf SVM model\n")
        pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
        svc = SVC(kernel='rbf', class_weight='balanced')
        model = make_pipeline(pca, svc)
        return model

    def get_best_parameters(self, pipeline_model):
        print("Exploring combinations of C and gamma parameters to determine the best parameters for our pipeline model...")
        param_grid = {
            'svc__C': [0.001, 0.01, 0.02, 0.1, 0.2, 1, 5, 10, 50, 100, 1000],
            'svc__gamma': [0.001, 0.01, 0.5, 1, 5, 10, 100, 1000]
        }
        search = GridSearchCV(pipeline_model, param_grid, n_jobs=-1)
        search.fit(X_train, y_train) #GridSearch only on training data to avoid leakage to test data
        print("Best parameter (CV score=%0.4f):" % search.best_score_) #Best Cross validation score
        print(search.best_params_) #These are the best parameters for C and gamma for our rbf SVC model
        return search.best_params_

    def set_and_train(self, pipeline_model, best_parameters, X_train, y_train):
        print("Setting the model pipeline's SVC params with best params as found by grid search...")
        pipeline_model.set_params(**best_parameters)

        print("Fitting the model pipeline (PCA and rbf SVC) with training data/label...\n")
        pipeline_model.fit(X_train, y_train)

        return pipeline_model

    def predict(self, pipeline_model, X_test):
        print("Making model predictions...\n")
        predictions = pipeline_model.predict(X_test)

        return predictions

    def print_classification_report(self, y_test, predictions):
        print("Printing classification report...")
        print(classification_report(y_test, predictions, target_names=self.people))

    def print_scores(self, y_test, predictions):
        print("Printing scores...")
        accuracy = accuracy_score(y_test, predictions)
        precision,recall,fscore,s=score(y_test,predictions, average='macro')
        p,r,f,support=score(y_test,predictions)
        print('Precision : {}'.format(precision))
        print('Recall    : {}'.format(recall))
        print('F-score   : {}'.format(fscore))
        print('Support   : {}'.format(support))
        print('Accuracy  : {}'.format(accuracy))
        print("\n")

    def print_24_test_images(self, X_test, y_test, predictions):
        print("Printing 24 (4x6 subplot) of random test images...")
        print("Title of image is the true label")
        print("Red text indicates this label has been misclassified\n")
        plt.figure(figsize=(15,15))

        for i in range(24): 
            ax = plt.subplot(4, 6, i + 1)
            random_int = random.randint(0, X_test.shape[0] - 1) #get random int within bounds of X_test
            chosen_image = X_test[random_int] #get random X_test image
            reshaped = chosen_image.reshape((62,47)) #reshape to image dimensions
            plt.imshow(reshaped) #show image
            actual_label = y_test[random_int] #this is the actual label
            predicted_label = predictions[random_int] #predicted label
            person = self.people[actual_label] #Print out actual person as title
            if(actual_label == predicted_label): #if actual label matches predicted label then text is black
                plt.title(person, color='k')
            else:
                plt.title(person, color='r') #otherwise text is red
            plt.axis("off")
        
        plt.show(block=True)

    def plot_confusion_matrix(self, y_test, predictions):
        print("Plotting confusion matrix heatmap for our test data results...")

        #plot confusion matrix matrix heatmap for test data predictions. X-axis is prediction. Y-axis is actual.
        cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7])

        plt.figure(figsize = (12, 8))

        # confusion matrix sns heatmap 
        ax = plt.axes()

        sns.heatmap(cm, annot=True, annot_kws={"size": 10}, fmt='d',cmap="Blues", ax = ax )
        ax.set_title('Confusion Matrix')

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.yticks(rotation=0)

        plt.show(block=True)
        print("Label number corresponds to index of people list")

        for count, value in enumerate(self.people):
            print(count, value)
    
if __name__ == "__main__":
    peopleImageClassifier = PeopleImageClassifier() #construct object instance of class
    peopleImageClassifier.load_data() #Load the data

    peopleImageClassifier.explore_data() #Explore the data

    peopleImageClassifier.print_24_data_images() #Print 24 images of full dataset

    X_train,X_test,y_train,y_test = peopleImageClassifier.split() #train_test_split 80:20 ratio

    X_train,X_test = peopleImageClassifier.simple_normalize(X_train, X_test) #Divide data by 255 to make each X input [0,1]

    pipeline_model = peopleImageClassifier.make_pipeline_model() #Make the desired PCA/SVC pipeline model

    best_parameters = peopleImageClassifier.get_best_parameters(pipeline_model) #Get the best parameters for C and gamma for our rbf SVC model via grid search cross-validation

    pipeline_model = peopleImageClassifier.set_and_train(pipeline_model, best_parameters, X_train, y_train) #Set the model with found best parameters and train with X_train, y_train
    
    predictions = peopleImageClassifier.predict(pipeline_model, X_test) #Get our model predictions for test data
    
    #Print out classification_report to show precision, recall, f1-score, and support for each label
    #As well as accuracy score and averages
    peopleImageClassifier.print_classification_report(y_test, predictions) 

    #print out precision, recall, f1-score as macro averages, and support list
    peopleImageClassifier.print_scores(y_test, predictions)

    # Plot 24 random test images as 4x6 subplot
    peopleImageClassifier.print_24_test_images(X_test, y_test, predictions)

    # Plot the confusion matrix heatmap for test data predictions. X-axis is prediction. Y-axis is actual.
    peopleImageClassifier.plot_confusion_matrix(y_test, predictions)


    
    
