import numpy
import random
import time
import sklearn
import pandas
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Sets random seed to increase repeatability
random.seed(23)
numpy.random.seed(23)


def current_ms() -> int:
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)

if __name__ == "__main__":
    """
    Main of the monitor
    """
    #0 load dataset PANDAS/NUMPY
    my_dataset = pandas.read_csv("./labelled_dataset.csv")
    label_obj = my_dataset["label"]
    data_obj = my_dataset.drop(columns=["label", "time", "datetime"]) 


    #1 SPLIT DATASET
    train_data, test_data, train_label, test_label = \
          train_test_split(data_obj, label_obj, test_size=0.5)
    
    # Set of classifiers that I want to run and compare
    classifiers = [VotingClassifier(estimators=[('lda', LinearDiscriminantAnalysis()),
                                                ('nb', GaussianNB()),
                                                ('dt', DecisionTreeClassifier())]),
                   StackingClassifier(estimators=[('lda', LinearDiscriminantAnalysis()),
                                                  ('nb', GaussianNB()),
                                                  ('dt', DecisionTreeClassifier())],
                                      final_estimator=RandomForestClassifier(n_estimators=10)),
                   DecisionTreeClassifier(),
                   GaussianNB(),
                   LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(n_neighbors=11),
                   RandomForestClassifier(n_estimators=10),
                   RandomForestClassifier(n_estimators=3),
                   GradientBoostingClassifier()]

      #3 TRAIN CLASSIFIER
    for clf in classifiers:
        # Training an algorithm
        before_train = current_ms()
        clf = clf.fit(train_data, train_label)
        after_train = current_ms()

        # Testing the trained model
        predicted_labels = clf.predict(test_data)
        end = current_ms()

        # Computing metrics to understand how good an algorithm is
        accuracy = sklearn.metrics.accuracy_score(test_label, predicted_labels)
        tn, fp, fn, tp = confusion_matrix(test_label, predicted_labels).ravel()
        print("%s: Accuracy is %.4f, train time: %d, test time: %d TP: %d, TN: %d, FN: %d, FP: %d" % (
            clf.__class__.__name__, accuracy, after_train - before_train, end - after_train, tp, tn, fn, fp))