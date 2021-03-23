import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head().to_markdown())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self, feature_cols):
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self, feature_cols, C, test_size, random_state):
        # split X and y into training and testing sets
        X, y = self.define_feature(feature_cols)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(C=C)
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self, feature_cols, C, test_size, random_state):
        model = self.train(feature_cols, C, test_size, random_state)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    print("Running Baseline training and prediction with LogisticRegression...\n")
    print("feature_cols = ['pregnant', 'insulin', 'bmi', 'age'] ; C = 1.0 ; test_size = 0.25 ; random_state = 0\n")
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
    C = 1.0
    test_size = 0.25
    random_state = 0
    classifer0 = DiabetesClassifier()
    result = classifer0.predict(feature_cols, C, test_size, random_state)
    print(f"Predicition={result}")
    score0 = classifer0.calculate_accuracy(result)
    print(f"score={score0}")
    con_matrix0 = classifer0.confusion_matrix(result)
    print(f"confusion_matrix=\n{con_matrix0}")

    print("\nRunning Solution1 training and prediction with LogisticRegression...\n")
    print("feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 1.0 ; test_size = 0.25 ; random_state = 0")
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    C = 1.0
    test_size = 0.25
    random_state = 0
    classifer1 = DiabetesClassifier()
    result = classifer1.predict(feature_cols, C, test_size, random_state)
    print(f"Predicition={result}")
    score1 = classifer1.calculate_accuracy(result)
    print(f"score={score1}")
    con_matrix1 = classifer1.confusion_matrix(result)
    print(f"confusion_matrix=\n{con_matrix1}")

    print("\nRunning Solution2 training and prediction with LogisticRegression...\n")
    print("feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 1.0 ; test_size = 0.20 ; random_state = 0")
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    C = 1.0
    test_size = 0.20
    random_state = 0
    classifer2 = DiabetesClassifier()
    result = classifer2.predict(feature_cols, C, test_size, random_state)
    print(f"Predicition={result}")
    score2 = classifer2.calculate_accuracy(result)
    print(f"score={score2}")
    con_matrix2 = classifer2.confusion_matrix(result)
    print(f"confusion_matrix=\n{con_matrix2}")

    print("\nRunning Solution3 training and prediction with LogisticRegression...\n")
    print("feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 2.0 ; test_size = 0.20 ; random_state = 0")
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    C = 2
    test_size = 0.20
    random_state = 0
    classifer3 = DiabetesClassifier()
    result = classifer3.predict(feature_cols, C, test_size, random_state)
    print(f"Predicition={result}")
    score3 = classifer3.calculate_accuracy(result)
    print(f"score={score3}")
    con_matrix3 = classifer3.confusion_matrix(result)
    print(f"confusion_matrix=\n{con_matrix3}")

    print("\nRunning Solution4 training and prediction with LogisticRegression...\n")
    print("feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'] ; C = 1.0 ; test_size = 0.20 ; random_state = 5")
    feature_cols = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    C = 1.0
    test_size = 0.20
    random_state = 5
    classifer4 = DiabetesClassifier()
    result = classifer4.predict(feature_cols, C, test_size, random_state)
    print(f"Predicition={result}")
    score4 = classifer4.calculate_accuracy(result)
    print(f"score={score4}")
    con_matrix4 = classifer4.confusion_matrix(result)
    print(f"confusion_matrix=\n{con_matrix4}")

    print("\nNow, showing the experiment results. See README.md for desired output with comments.")

    con_matrix0_N = con_matrix0[0]
    con_matrix0_P = con_matrix0[1]

    con_matrix1_N = con_matrix1[0]
    con_matrix1_P = con_matrix1[1]

    con_matrix2_N = con_matrix2[0]
    con_matrix2_P = con_matrix2[1]

    con_matrix3_N = con_matrix3[0]
    con_matrix3_P = con_matrix3[1]

    con_matrix4_N = con_matrix4[0]
    con_matrix4_P = con_matrix4[1]

    print("Experiment | Accuracy           | Confusion Matrix ")
    print(f"Baseline   | {score0} | [{con_matrix0_N} {con_matrix0_P}]")
    print(f"Solution 1 | {score1} | [{con_matrix1_N} {con_matrix1_P}]")
    print(f"Solution 2 | {score2} | [{con_matrix2_N} {con_matrix2_P}]")
    print(f"Solution 3 | {score3} | [{con_matrix3_N} {con_matrix3_P}]")
    print(f"Solution 4 | {score4} | [{con_matrix4_N} {con_matrix4_P}]")


    
    
