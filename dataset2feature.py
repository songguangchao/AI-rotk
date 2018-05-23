import numpy as np
import pandas as pd

data0X = pd.read_csv('X-18000-0.csv')
column_names = ['matchID', 'winner', 'hero1','hero2', 'hero3','hero4','hero5','hero6','hero7',
            'hero8','hero9','hero10',]
data0y = pd.read_csv('y-18000-0.csv')
data1 = pd.read_csv('dotaMatch.csv', names=column_names)

data1X = data1.iloc[:,2:]
data1X_array = data1X.values
data1y = data1.iloc[:,1:2]
data1y.replace(-1,0,inplace=True)
data1y.rename(columns={"winner": '0'}, inplace = True)

delete_list  = []

for i in range(np.shape(data1X_array)[0]):
    if np.sum(data1X_array[i]) < 10:
        delete_list.append(i)
    for j in range(10):
        if data1X_array[i][j] >107:
            delete_list.append(i)
            break
        
data1X.drop(delete_list, inplace=True)
data1y.drop(delete_list, inplace=True)


data1X_array = data1X.values
data1X_onehot = np.zeros((np.shape(data1X)[0], 216))

for i in range(np.shape(data1X_array)[0]):
    for j in range(5):
        data1X_onehot[i][data1X_array[i][j]] = 1
        data1X_onehot[i][data1X_array[i][j+5] + 108] = 1

column_onehot = []

for i in range(216):
    column_onehot.append(str(i))
    
data1X = pd.DataFrame(data1X_onehot, columns = column_onehot)
data_X = pd.concat([data0X, data1X], ignore_index=True)
data_y = pd.concat([data0y, data1y], ignore_index=True)
print(np.shape(data_X))
data_X.drop(['0','108'],axis=1, inplace=True)
print(np.shape(data_X))
data_X.to_csv('dataset_X.csv',index=False)
data_y.to_csv('dataset_y.csv',index=False)

print('end')


