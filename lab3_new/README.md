# Lab 3 - Classification

[Pima Indian Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv) from the UCI Machine Learning Repository

## Question

* Can we predict the diabetes status of a patient given their health measurements? Build a classifer and calculate Confusion matrix with

- True Positives (TP): we correctly predicted that they do have diabetes
- True Negatives (TN): we correctly predicted that they don't have diabetes
- False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")
- False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")

-----------------------------

## Student Notes and Desired Output
The full output of my experiment can be seen in output.txt. I ran 4 separate experiments aside from the baseline with various changes to which feature columns to use / include for training and testing the model, the value of the regularization parameter C in LogisticRegression, and the value of the test_size and random_state of sklearn train_test_split function.

The desired output is shown below:
| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] | feature_cols = ['pregnant', 'insulin', 'bmi', 'age'] ; C = 1.0 ; test_size = 0.25 ; random_state = 0
| Solution 1   | 0.7916666666666666 | [[115  15] [25 37]] |  feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 1.0 ; test_size = 0.25 ; random_state = 0 |
| Solution 2   | 0.8246753246753247  | [[98  9] [18 29]] |  feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 1.0 ; test_size = 0.20 ; random_state = 0 |
| Solution 3   | 0.7922077922077922  | [[95 12] [20 27]] |  feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 2.0 ; test_size = 0.20 ; random_state = 0 |
| Solution 4   | 0.7857142857142857  | [[87 13] [20 34]] |  feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 1.0 ; test_size = 0.20 ; random_state = 5 |

To explain, in Solution1, I decided to use all of the features to train the LogisticRegression model and test the test data. As we can see, the accuracy improved dramatically (up 12% from the baseline that only used pregnant, insulin, bmi, and age as features). We can see that the confusion matrix now has less false negatives (almost halved from baseline) and also more true positives (37 versus 16 of baseline). 

In Solution2, I decided to change the test size from 0.25 (default) to 0.20 so that more samples can be used for training (but tradeoff is that there will be less data samples for testing). This improved our accuracy a bit (up 3 percent from Solution1). I still used all the features as I believe they are all important. As we can see from the confusion matrix, the number of false negatives is 18 and the number of false positives is just 9 (however, we need to take into consideration that the test size is now smaller, and there are only 107 negative and 47 positive samples).

In Solution3, I decided to use Solution2 but instead of the regularization parameter of LogisticRegression set to 1.0, I used C=2.0 (more regularization, limits the "slope" of the model. Introduce a bit of bias so that the variance is lower). As we can see, this regularization added a bit too much bias as we can see, the accuracy decreased 3% from Solution2. The confusion matrix shows that there are now more false positives (12 vs 9 of solution2).

In Solution4, I decided to use our best Solution2, but changed the random state of train_test_split to 5 (instead of 0). As we can see, this random state lowered the accuracy by 3% (to 78%). This shows that how we select samples in training and testing from the dataset can result in different results. This is why cross validation is important to confirming whether or not your model is the best.






