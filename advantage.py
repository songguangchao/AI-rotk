import numpy as np
import pandas as pd

def onehot2Normal(data_X,herosNum):
    rowNum = np.shape(data_X)[0]
    columnNum = np.shape(data_X)[1]
    data_X_normal = np.zeros((rowNum, 10))
    
    for i in range(rowNum):
        cnt = 0
        for j in range(columnNum):
            if int(data_X[i][j]) == 1:
                if j <= herosNum - 1:
                    data_X_normal[i][cnt] = j
                    cnt += 1
                else:
                    data_X_normal[i][cnt] = j - herosNum
                    cnt += 1
                    if cnt == 10:
                        break
    
    return data_X_normal        
        
def synergyAndCounter(data_X, data_y, herosNum):
    
    rowNum = np.shape(data_X)[0]
    
    ally = np.zeros((herosNum, herosNum, 2))
    enemy = np.zeros((herosNum, herosNum, 2))
    
    for i in range(rowNum):
        if int(data_y[i][0]) == 1:
            for j in range(5):
                Radient_hero1 = int(data_X[i][j])

                Dire_hero1 = int(data_X[i][j+5])
                for k in range(j+1,5):

                    Radient_hero2 = int(data_X[i][k])

                    Dire_hero2 = int(data_X[i][k+5])
                    ally[Radient_hero1][Radient_hero2][0] += 1
                    ally[Radient_hero2][Radient_hero1][0] += 1
                    ally[Dire_hero1][Dire_hero2][1] += 1
                    ally[Dire_hero2][Dire_hero1][1] += 1
                    enemy[Radient_hero1][Dire_hero1][0] += 1
                    enemy[Dire_hero1][Radient_hero1][1] += 1
                    enemy[Radient_hero1][Dire_hero2][0] += 1
                    enemy[Dire_hero2][Radient_hero1][1] += 1
                    enemy[Radient_hero2][Dire_hero1][0] += 1
                    enemy[Dire_hero1][Radient_hero2][1] += 1
                    enemy[Radient_hero2][Dire_hero2][0] += 1
                    enemy[Dire_hero2][Radient_hero2][1] += 1
        else:
            for j in range(5):
                Radient_hero1 = int(data_X[i][j])
                Dire_hero1 = int(data_X[i][j+5])
                for k in range(j+1,5):

                    Radient_hero2 = int(data_X[i][k])
                    Dire_hero2 = int(data_X[i][k+5])
                    ally[Radient_hero1][Radient_hero2][1] += 1
                    ally[Radient_hero2][Radient_hero1][1] += 1
                    ally[Dire_hero1][Dire_hero2][0] += 1
                    ally[Dire_hero2][Dire_hero1][0] += 1
                    enemy[Radient_hero1][Dire_hero1][1] += 1
                    enemy[Dire_hero1][Radient_hero1][0] += 1
                    enemy[Radient_hero1][Dire_hero2][1] += 1
                    enemy[Dire_hero2][Radient_hero1][0] += 1
                    enemy[Radient_hero2][Dire_hero1][1] += 1
                    enemy[Dire_hero1][Radient_hero2][0] += 1
                    enemy[Radient_hero2][Dire_hero2][1] += 1
                    enemy[Dire_hero2][Radient_hero2][0] += 1
                    
    synergy = np.zeros((herosNum,herosNum))

    for i in range(herosNum):
        for j in range(herosNum):
            if ally[i][j][0] + ally[i][j][1] != 0:
                synergy[i][j] = ally[i][j][0] / (ally[i][j][0] + ally[i][j][1])
    counter = np.zeros((herosNum,herosNum))
          
    for i in range(herosNum):
        for j in range(herosNum):
            if enemy[i][j][0] + enemy[i][j][1] != 0:
                counter[i][j] = enemy[i][j][0] / (enemy[i][j][0] + enemy[i][j][1])
    return synergy,counter
                

def main():
    herosNum = 107
    data_X = pd.read_csv('dataset_X.csv')
    data_y = pd.read_csv('dataset_y.csv')
    data_X = data_X.values
    data_y = data_y.values
    data_X = onehot2Normal(data_X,herosNum)
    synergy, counter = synergyAndCounter(data_X, data_y, herosNum)
    synergy_df = pd.DataFrame(synergy)
    synergy_df.to_csv('synergy.csv',index=False)
    counter_df = pd.DataFrame(counter)
    counter_df.to_csv('counter.csv',index=False)
    
    print('end')
if __name__ == '__main__':
	main()
                
        