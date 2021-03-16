# Lab 3

Try to build an even-number detector classifier for the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset from Scikit-Learn. You can use either binary classifer or multi-classifier.

Use tips.py and [How to load MINIST Dataset](http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/)

---------

## Student Notes
First, I downloaded and used the MNIST dataset (mnist_train.csv) found on Kaggle here: [Kaggle Dataset](https://www.kaggle.com/oddrationale/mnist-in-csv). Please make sure this file is in the same directory as my lab3.py file in order to run it.

For the save_fig method, in the os.path.join, I set the parameter to be fig_id + ".png". So, we do not care about the project root directory nor the image directory, the image will just be saved in the current directory with the name that is passed into the method.

I made the random_digit method randomly choose a random entry in the dataframe that is passed into it. It sets the random seed based on the current time, and makes sure to choose a random_entry that is within the bounds of the dataframe indices. It saves the image (handwritten number) of the selected random entry in the file some_digit_plot.png. It returns but the random digit pixel data (array of pixel values), as well as the index it was located in the dataframe.

We do not use the load_and_sort method because because it is fetching the mnist dataset from sklearn. Instead, we use a local copy of the mnist dataset.

We do not use sort_by_target because we use train_test_split to randomly shuffle the MNIST training set into training and test set.

In train_model method, we convert the y_train series (labels are 0-9) into a y_train_4 series (1 for a 4, 0 if not a 4). Then, we use the SGDClassifier model and fit X_train and y_train_4 to fit it. The method returns the model.

In the predict method, which takes in a random entry pixel data (array) and the model, we get the prediction of the model on this random entry. In this case, our model will be the SGDClassifier trained to identify 4 or not. If prediction is true, it outputs that the model has predicted the digit to be a 4.

In the calculate_cross_val_score method, which accepts X (features) and y (target) of the whole MNIST train dataset, we calculate the cross_val_score on the SGDClassifier using 3 different cross_validation splits (cv=3). We make sure to convert the y (target) column to be 1 or 0 based on if the label is 4 before we pass it into cross_val_score because our model is a binary classification model on the number 4.

In the test method, we run the program of this lab assignment. We load the dataset and split the dataset. We train the SGDClassification model using X_train and y_train. We then select a random entry from the test set using random_digit method. And then with the random digit data, we predict what the model thinks of the digit (if it is or is not a 4). We also print out whether the prediction is correct or not (comparing it to the actual number). Finally, we call calculate_cross_val_score to check how this model performs on the whole dataset using 3 fold cross validation. The accuracy scores were very high (0.97 mean accuracy and a std dev close to 0).

lab3_output_2.txt shows complete output of my work, and includes the calling of the method calculate_cross_val_score. In this run, the selected random entry was labeled 5, and we can see that the model predicted that the digit was not a 4 (correct).

I ran the program a few more times in order for the selected random entry to be a 4 to see if the model predicts it correctly. lab3_output.txt shows this scenario, and the model indeed predicted that the digit is a 4. (For these runs, I did not call method calculate_cross_val_score because it takes a while to complete. Results of calculate_cross_val_score can be seen in lab3_output_2.txt)







