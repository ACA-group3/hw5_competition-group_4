#We should test best Features Here
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import pandas as pd

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
            aux = tpr
            tpr = fpr
            fpr = aux
        plt.plot(fpr, tpr, label=data.columns[col] + ', auc = ' \
                                 + str(np.round(auc(fpr, tpr), decimals=3)))
    plt.title('ROC curve - Iris features')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    pass
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
    return data
data = removeNullColumns(data,"In-hospital_death")
data= replaceNullValues(data)