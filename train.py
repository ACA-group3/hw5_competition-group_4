
import pyforest
import sys
from impyute.imputation.cs import fast_knn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
import pickle
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('Survival_dataset.csv')
df= df.drop(['Survival', 'Length_of_stay'], axis=1)
imputed_training=fast_knn(df.values, k=5)
df_filled_kNN = pd.DataFrame(data = imputed_training, columns = df.columns)
x = df_filled_kNN[df_filled_kNN.columns.difference(['In-hospital_death'])]
y = df_filled_kNN['In-hospital_death']
#small test of feature importance with decision tree
model = DecisionTreeClassifier()
model.fit(x, y)
importance = model.feature_importances_
imp_zip = zip(np.arange(0, len(importance), 1), importance)
importance_sorted = sorted(imp_zip, key = lambda x: x[1], reverse=True)
selected_features_indxs = [i for i, j in importance_sorted if j>0.005]
filtered_x = x.iloc[:, selected_features_indxs]
lr = LogisticRegression()
lr.fit(x, y)
y_pred_lr = lr.predict(x)
lda = LinearDiscriminantAnalysis()
lda.fit(x, y)
y_pred_lda = lda.predict(x)
qda = QuadraticDiscriminantAnalysis()
qda.fit(x, y)
y_pred_qda = qda.predict(x)
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(x, y)
y_pred_dt = dt.predict(x)
nb = GaussianNB()
nb.fit(x, y)
y_pred_nb = nb.predict(x)
svm = SVC()
svm.fit(x, y)
y_pred_svm = nb.predict(x)
cl_report_lr = classification_report(y, y_pred_lr, output_dict=True)
cl_report_lda = classification_report(y, y_pred_lda, output_dict=True)
cl_report_qda = classification_report(y, y_pred_qda, output_dict=True)
cl_report_dt = classification_report(y, y_pred_dt, output_dict=True)
cl_report_nb = classification_report(y, y_pred_nb, output_dict=True)
cl_report_svm = classification_report(y, y_pred_svm, output_dict=True)
scores = {"Logistic Regression" : {"precision" : cl_report_lr['weighted avg']['precision'],
                                   "recall" :  cl_report_lr['weighted avg']['recall'],
                                   "f1_score" :  cl_report_lr['weighted avg']['f1-score']},
          "LDA" : {"precision" : cl_report_lda['weighted avg']['precision'],
                                   "recall" :  cl_report_lda['weighted avg']['recall'],
                                   "f1_score" :  cl_report_lda['weighted avg']['f1-score']},
          "QDA" : {"precision" : cl_report_qda['weighted avg']['precision'],
                                   "recall" :  cl_report_qda['weighted avg']['recall'],
                                   "f1_score" :  cl_report_qda['weighted avg']['f1-score']},
          "Decision Tree" : {"precision" : cl_report_dt['weighted avg']['precision'],
                                   "recall" :  cl_report_dt['weighted avg']['recall'],
                                   "f1_score" :  cl_report_dt['weighted avg']['f1-score']},
          "Naive Bayes" : {"precision" : cl_report_nb['weighted avg']['precision'],
                                   "recall" :  cl_report_nb['weighted avg']['recall'],
                                   "f1_score" :  cl_report_nb['weighted avg']['f1-score']},
          "SVM" : {"precision" : cl_report_svm['weighted avg']['precision'],
                                   "recall" :  cl_report_svm['weighted avg']['recall'],
                                   "f1_score" :  cl_report_svm['weighted avg']['f1-score']}
    
}
scores_table = pd.DataFrame(scores).T
#trying to get more generalized scores with cross validation
LR = LogisticRegression(random_state=42)
LDA = LinearDiscriminantAnalysis()
QDA = QuadraticDiscriminantAnalysis()
DT = DecisionTreeClassifier(max_depth=4, random_state=42)
NB = GaussianNB()
SVM = SVC()
Bagging_SVM = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)
Random_forest = RandomForestClassifier(random_state = 42, max_depth = 3)
AdaBoost = AdaBoostClassifier(base_estimator=LogisticRegression(),
                        n_estimators=10, random_state=0, algorithm="SAMME")
cv = KFold(n_splits=10, shuffle=True, random_state=142)
scoring = ['precision', 'recall', 'f1', 'roc_auc']
LR_scores = cross_validate(LR,x, y, scoring=scoring, cv=cv)
LDA_scores = cross_validate(LDA, x, y,scoring=scoring, cv=cv)
QDA_scores = cross_validate(QDA, x, y,scoring=scoring, cv=cv)
DT_scores = cross_validate(DT, x, y,scoring=scoring, cv=cv)
NB_scores = cross_validate(NB, x, y,scoring=scoring, cv=cv)
SVM_scores = cross_validate(SVM, x, y, scoring=scoring,cv=cv)
Bagging_scores = cross_validate(Bagging_SVM, x, y, scoring=scoring, cv=cv)
Random_forest_scores = cross_validate(Random_forest, x, y, scoring=scoring, cv=cv)
AdaBoost_scores = cross_validate(AdaBoost, x, y, scoring=scoring, cv=cv)
scores_cv = {"Logistic Regression" : {"precision" : np.mean(LR_scores['test_precision_weighted']),
                                   "recall" :  np.mean(LR_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(LR_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(LR_scores['test_roc_auc'])},
        "LDA" : {"precision" : np.mean(LDA_scores['test_precision_weighted']),
                                   "recall" :  np.mean(LDA_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(LDA_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(LDA_scores['test_roc_auc'])},
        "QDA" : {"precision" : np.mean(QDA_scores['test_precision_weighted']),
                                   "recall" :  np.mean(QDA_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(QDA_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(QDA_scores['test_roc_auc'])},
        "Decision tree" : {"precision" : np.mean(DT_scores['test_precision_weighted']),
                                   "recall" :  np.mean(DT_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(DT_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(DT_scores['test_roc_auc'])},
        "Naive Bayes" : {"precision" : np.mean(NB_scores['test_precision_weighted']),
                                   "recall" :  np.mean(NB_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(NB_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(NB_scores['test_roc_auc'])},
        "SVM" : {"precision" : np.mean(SVM_scores['test_precision_weighted']),
                                   "recall" :  np.mean(SVM_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(SVM_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(SVM_scores['test_roc_auc'])},
        "Bagging_SVM" : {"precision" : np.mean(Bagging_scores['test_precision_weighted']),
                                   "recall" :  np.mean(Bagging_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(Bagging_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(Bagging_scores['test_roc_auc'])},
        "Random_forest" : {"precision" : np.mean(Random_forest_scores['test_precision_weighted']),
                                   "recall" :  np.mean(Random_forest_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(Random_forest_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(Random_forest_scores['test_roc_auc'])},
        "AdaBoost" : {"precision" : np.mean(AdaBoost_scores['test_precision_weighted']),
                                   "recall" :  np.mean(AdaBoost_scores['test_recall_weighted']),
                                   "f1_score" :  np.mean(AdaBoost_scores['test_f1_weighted']),
                                    "AUC" :  np.mean(AdaBoost_scores['test_roc_auc'])},
}
scores_cv_table = pd.DataFrame(scores_cv).T
print(scores_cv_table)




