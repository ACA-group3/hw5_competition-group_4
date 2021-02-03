#We should test best Features Here
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
from sklearn import svm
from sklearn.model_selection import cross_val_score

# loading dataset
data = pd.read_csv("Survival_dataset.csv")


def ROC_Curve(data,target):
    x, y = data.loc[:, data.columns != target], data[target]
    y_ = y == 1
    plt.figure(figsize=(13, 7))
    for col in range(x.shape[1]):
        tpr, fpr = [], []
        min(x.iloc[:, col])
        max(x.iloc[:, col])

        for threshold in np.linspace(min(x.iloc[:, col]), max(x.iloc[:, col]), 100):
            detP = x.iloc[:, col] < threshold
            tpr.append(sum(detP & y_) / sum(y_))  # TP/P, aka recall
            fpr.append(sum(detP & (~y_)) / sum((~y_)))  # FP/N

        if auc(fpr, tpr) < .5:
            continue
        plt.plot(fpr, tpr, label=data.columns[col] + ', auc = ' \
                                 + str(np.round(auc(fpr, tpr), decimals=3)))
    plt.title('ROC curve - Iris features')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    pass

def tryAllFeatures(data,target):
    x, y = data.loc[:, data.columns != target], data[target]
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    max_value = 0
    max_columns = itertools
    clf = svm.SVC(random_state=0)
    for i in range(X_train.size):
        columns = itertools.permutations(x.columns,i)
        new_data=pd.DataFrame
        for j in columns:
            temp = 0
            for k in j:
                new_data.append(x[k])

            if(temp>max_value):
                max_columns = columns
    return max_columns
def removeNullColumns(data,target):
    x, y = data.loc[:, data.columns != target], data[target]
    for col in x.columns:
        percetnage = data[col].isnull().sum()/len(data[col])*100
        if(percetnage>=60):
            del data[col]
    return data
def replaceNullValues(data):
    for col in data.columns:
        data[col].fillna(data[col].median())
        print(data[col].isnull().sum())
    return data
data = removeNullColumns(data,"In-hospital_death")
data= replaceNullValues(data)
print(data.info())
tryAllFeatures(data,"In-hospital_death")