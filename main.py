import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import KFold


def main():
    task_1()
    plt.show()


def task_1():
    iris = load_iris()
    X = iris.data
    y = iris.target

    kf = KFold(n_splits=len(iris.target))
    avg_acc_train = []
    avg_acc_test = []

    for i in range(1, 6):
        # create decision tree classifier object
        clf = DecisionTreeClassifier(max_depth=i)
        acc_train = []
        acc_test = []

        for train_index, test_index in kf.split(iris.data):
            # train classifier
            clf = clf.fit(iris.data[train_index], iris.target[train_index])

            # predict response for test data
            prediction_train = clf.predict(iris.data[train_index])
            prediction_test = clf.predict(iris.data[test_index])
            acc_train.append(metrics.accuracy_score(iris.target[train_index], prediction_train))
            acc_test.append(metrics.accuracy_score(iris.target[test_index], prediction_test))
        avg_acc_train.append(np.mean(acc_train))
        avg_acc_test.append(np.mean(acc_test))

    plt.figure()
    plt.plot(np.arange(len(avg_acc_test)) + 1, avg_acc_test)
    plt.plot(np.arange(len(avg_acc_train)) + 1, avg_acc_train)

    best_depth = np.argmax(avg_acc_test) + 1
    print("Best depth:", best_depth)


main()
