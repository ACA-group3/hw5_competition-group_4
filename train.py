# we should write our final steps here
#import Model
import  sys
from impyute.imputation.cs import fast_knn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold,cross_validate
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier


def load_CSV(path):
    return pd.read_csv(path)

def fill_NA(df):
    imputed_training = fast_knn(df.values, k=5)
    return pd.DataFrame(data=imputed_training,columns=df.columns)

def preiction_report(x,y):
    models=[LogisticRegression(),LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
            DecisionTreeClassifier(),GaussianNB(),SVC(),BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0),
            RandomForestClassifier(random_state=42, max_depth=3),AdaBoostClassifier(base_estimator=LogisticRegression(),
                                  n_estimators=10, random_state=0, algorithm="SAMME")]
    scores = []
    cv = KFold(n_splits=10, shuffle=True, random_state=142)
    scoring = ['precision', 'recall', 'f1', 'roc_auc']
    for model in models:
        scores.append((cross_validate(model, x, y, scoring=scoring, cv=cv)))

    scores_cv = {"Logistic Regression": {"precision": np.mean(scores[0]['test_precision']),
                                     "recall": np.mean(scores[0]['test_recall']),
                                     "f1_score": np.mean(scores[0]['test_f1']),
                                     "AUC": np.mean(scores[0]['test_roc_auc'])},
             "LDA": {"precision": np.mean(scores[1]['test_precision']),
                     "recall": np.mean(scores[1]['test_recall']),
                     "f1_score": np.mean(scores[1]['test_f1']),
                     "AUC": np.mean(scores[1]['test_roc_auc'])},
             "QDA": {"precision": np.mean(scores[2]['test_precision']),
                     "recall": np.mean(scores[2]['test_recall']),
                     "f1_score": np.mean(scores[2]['test_f1']),
                     "AUC": np.mean(scores[2]['test_roc_auc'])},
             "Decision tree": {"precision": np.mean(scores[3]['test_precision']),
                               "recall": np.mean(scores[3]['test_recall']),
                               "f1_score": np.mean(scores[3]['test_f1']),
                               "AUC": np.mean(scores[3]['test_roc_auc'])},
             "Naive Bayes": {"precision": np.mean(scores[4]['test_precision']),
                             "recall": np.mean(scores[4]['test_recall']),
                             "f1_score": np.mean(scores[4]['test_f1']),
                             "AUC": np.mean(scores[4]['test_roc_auc'])},
             "SVM": {"precision": np.mean(scores[5]['test_precision']),
                     "recall": np.mean(scores[5]['test_recall']),
                     "f1_score": np.mean(scores[5]['test_f1']),
                     "AUC": np.mean(scores[5]['test_roc_auc'])},
             "Bagging_SVM": {"precision": np.mean(scores[6]['test_precision']),
                             "recall": np.mean(scores[6]['test_recall']),
                             "f1_score": np.mean(scores[6]['test_f1']),
                             "AUC": np.mean(scores[6]['test_roc_auc'])},
             "Random_forest": {"precision": np.mean(scores[7]['test_precision']),
                               "recall": np.mean(scores[7]['test_recall']),
                               "f1_score": np.mean(scores[7]['test_f1']),
                               "AUC": np.mean(scores[7]['test_roc_auc'])},
             "AdaBoost": {"precision": np.mean(scores[8]['test_precision']),
                          "recall": np.mean(scores[8]['test_recall']),
                          "f1_score": np.mean(scores[8]['test_f1']),
                          "AUC": np.mean(scores[8]['test_roc_auc'])},
             }
    scores_cv_table = pd.DataFrame(scores_cv).T
    print(scores_cv_table)
    best_model = get_Best_Model(models,scores)
    save_Best_Model(best_model,x,y)
    return scores_cv_table

def get_Best_Model(models,scores):
    max_val=0
    best_model = object
    for i in range(len(models)-1):
        score =[np.mean(scores[i]['test_precision']),
                              np.mean(scores[i]['test_recall']),
                              np.mean(scores[i]['test_f1']),
                              np.mean(scores[i]['test_roc_auc'])
                 ]
        mean_score = np.mean(score)
        if (mean_score>=max_val):
            max_val=mean_score
            best_model= models[i]
    return best_model

def save_Best_Model(model,x,y):
    model.fit(x, y)
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    pass

path = "Survival_dataset.csv"
target = 'In-hospital_death'
df = load_CSV(path)
df = df.drop(['Survival', 'Length_of_stay'], axis=1)
df = fill_NA(df)
x = df[df.columns.difference([target])]
y = df[target]
preiction_report(x,y)