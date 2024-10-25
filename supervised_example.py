import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

numpy.random.seed(23)
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

    #2 choose classifer SCIKIT LEARN
    clf= tree.DecisionTreeClassifier()

    #3 TRAIN CLASSIFIER
    clf = clf.fit(train_data, train_label)

    #4 test classifier
    predicted_label = clf.predict(test_data)
    accuracy = accuracy_score(test_label, predicted_label)
    print("Accuracy is %.6f" %accuracy)
    #tn, fp, fn, tp = confusion_matrix(test_label,predicted_label).ra
   # print("TP: %d, TN: %d, FN: %d, FP: %d" %(tp, tn, fn, fp))