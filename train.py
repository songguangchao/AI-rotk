import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from advantage import onehot2Normal
import seaborn as sns


def dataSet(herosNum, filePathX, filePathY, synergy = [], counter = []):
    #column_names = ['matchID', 'winner', 'hero1','hero2', 'hero3','hero4','hero5','hero6','hero7',
    #            'hero8','hero9','hero10',]
    #data = pd.read_csv(filePath, names=column_names)
    #data_heros = data.iloc[:,2:]
    #data_winner = data.iloc[:,1]
    #column_dummy = []
    data_heros = pd.read_csv(filePathX)
    data_winner = pd.read_csv(filePathY)
    #print(data.iloc[1,2:])
    """
    for i in range(herosNum):
        column_dummy.append('RadiantHero' + str(i))
        
        for i in range(herosNum):
            column_dummy.append('DireHero' + str(i))
           """
    #data_x = np.zeros((np.shape(data_heros)[0],herosNum * 2))
    #data_y = np.zeros(np.shape(data_winner)[0])
   # data_x = np.zeros((70000,herosNum * 2))
    #data_y = np.zeros(70000)
    #for i in range(np.shape(data_winner)[0]):
    #for i in range(70000):
    #    data_x[i][0] = 1
    #    for j in range(10):
     #       data_x[i][data_heros.iat[i,j]] = 1
     #   if data_winner[i] == 1:
      #      data_y[i] = 1
     #   else:
     #       data_y[i] = 0
   # return data_x, data_y
    data_x = data_heros.values
    data_y = data_winner.values
    
    if len(synergy):        
        data_X_normal = onehot2Normal(data_heros.values, herosNum)
        data_synergy = np.zeros((np.shape(data_heros.values)[0], 2))
        
        for i in range(np.shape(data_heros.values)[0]):
            Radiant_synergy = 0
            Dire_synergy = 0
            for j in range(5):
                for k in range(j+1, 5):
                    Radiant_synergy += synergy[int(data_X_normal[i][j])][int(data_X_normal[i][k])]
                    Dire_synergy += synergy[int(data_X_normal[i][j+5])][int(data_X_normal[i][k+5])]
            data_synergy[i][0] = Radiant_synergy
            data_synergy[i][1] = Dire_synergy
        data_synergy_df = pd.DataFrame(data_synergy)
        data_x = pd.concat([pd.DataFrame(data_x), data_synergy_df], axis=1, join='inner') 
        data_x = data_x.values
        print(np.shape(data_x))
        
        
    if len(counter):        
        data_X_normal = onehot2Normal(data_heros.values, herosNum)
        data_counter = np.zeros((np.shape(data_heros.values)[0], 2))
        
        for i in range(np.shape(data_heros.values)[0]):
            Radiant_counter = 0
            Dire_counter = 0
            for j in range(5):
                for k in range(j+1, 5):
                    Radiant_counter += counter[int(data_X_normal[i][j])][int(data_X_normal[i][k])]
                    Dire_counter += counter[int(data_X_normal[i][j+5])][int(data_X_normal[i][k+5])]
            data_counter[i][0] = Radiant_counter
            data_synergy[i][1] = Dire_counter
            
        data_counter_df = pd.DataFrame(data_counter)
        
        data_x = pd.concat([pd.DataFrame(data_x), data_counter_df], axis=1, join='inner') 
        data_x = data_x.values
        

    print(np.shape(data_x))
    
    return  data_x, data_y

def prediction(data_x, data_y):
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('trainnig start')
    """
    model = LogisticRegression(C=0.005, random_state=42, max_iter = 1000)
    scores = cross_val_score(model, X_train, y_train, cv = 7, scoring='roc_auc',
                                               n_jobs=1)
    scores_mean = np.mean(scores)
    model = LogisticRegression(C=0.005, random_state=42)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
    labels = model.predict(X_test)
    acc_score = accuracy_score(y_test, labels)
    """
    model = LogisticRegression(C=0.005, random_state=42)
    model.fit(X_train, y_train)
    scores=cross_val_score(model,X_train,y_train,cv=5)
    #score = model.score(X_test, y_test)
    """
    model_dict = {}
    model_dict['scaler'] = scaler
    model_dict['model'] = model
    
    joblib.dump(model_dict, "trained_model.m“)
    """
    score_mean = np.mean(scores)
    #return X_train.shape[0], X_test.shape[0], scores_mean, roc_auc, acc_score
    return score_mean
def main():
    herosNum = 107 
    filePathX = 'dataset_X.csv'
    filePathY = 'dataset_y.csv'
    synergy = pd.read_csv('synergy.csv')
    synergy = synergy.values
    counter = pd.read_csv('counter.csv')
    counter = counter.values
    matrix_x,matrix_y = dataSet(herosNum, filePathX, filePathY)
    print('data has been created')
    #_, _, cv_score, roc_auc, _ = prediction(matrix_x, matrix_y)

    score  = prediction(matrix_x[:18000], matrix_y[:18000])
    print(score)

if __name__ == '__main__':
	main()
