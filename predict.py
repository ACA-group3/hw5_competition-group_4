# we should implement our model here
import pickle
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from impyute.imputation.cs import fast_knn

def filterFeatures():
    #I should write it
    pass


def save_data(prob):
    print(prob)
    np.save('predictions', prob)


if __name__ == "__main__":
    opt,arg = sys.argv[1:2],sys.argv[2:3]

    if(opt[0] == '--test'):
        df=pd.read_csv(arg[0])
        df = df.drop(['Survival', 'Length_of_stay', 'In-hospital_death'], axis=1)
        with open('best_model.pkl', 'rb') as f:
           model = pickle.load(f)
        scaler = StandardScaler()
        data=fast_knn(df.values,k=5)
        predictions = model.predict_proba(data)
        save_data(predictions)



