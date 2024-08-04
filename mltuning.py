 
import time
def _dtw(output, target, window):
    n, m = len(output), len(target)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix += float("Inf")
    dtw_matrix[0, 0] = 0
    for i in range(1, n+1):
        a, b = np.max([1, i-w]), np.min([m, i+w])+1
        dtw_matrix[i,a:b] = 0
        
        
        for j in range(a, b):
            cost = np.abs(output[i-1] - target[j-1])
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix[-1, -1]



from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


params = {
    "kernel": ['rbf','linear','poly'],
    "C": [0.01, 0.1, 1, 10],
    "gamma":[0.0001,0.001,0.01],
    "degree":[1,2,3]
}
svr = SVR()
gssvr = GridSearchCV(svr, params, scoring = 'neg_mean_squared_error')

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
#pipe2 = Pipeline([('poly2', PolynomialFeatures(2)), ('linReg', lr)])
#pipe3 = Pipeline([('poly3', PolynomialFeatures(3)), ('linReg', lr)])

from sklearn.ensemble import RandomForestRegressor
params = {
    "n_estimators": [50,100,500]
    
}
rfg = RandomForestRegressor()
gsrfg = GridSearchCV(rfg, params, scoring = 'neg_mean_squared_error')

from sklearn.linear_model import BayesianRidge
params = {
    "max_iter": [100,1000,2000],
    "tol": [1e-6, 1e-8, 1e-10]
}
br = BayesianRidge()
gsbr= GridSearchCV(br, params, scoring = 'neg_mean_squared_error')

from sklearn.neighbors import KNeighborsRegressor
params = {
    "n_neighbors": [10, 20, 20, 50],
    "weights": ['uniform','distance'],
    "algorithm": ['ball_tree', 'kd_tree', 'brute']
    
}
knn = KNeighborsRegressor()
gsknn = GridSearchCV(knn, params, scoring = 'neg_mean_squared_error')




classifiers = {
    "Linear Regression": lr,
    "Bayesian Ridge Regression": gsbr,
    "Support Vector Regression": gssvr,
    "Random Forest Regression": gsrfg,
    "KNN-Regression": gsknn
}


from Scripts.utils import *  
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Scripts.FastDTW import *

columns = ['Date','Open','High','Low','Close','Volume','Yield','PercentageVolume',
           'SMA6','EMA6','WMA6','HMA6','SMA20','EMA20','WMA20','HMA20','SMA50','EMA50','WMA50','HMA50',
           'SMA100','EMA100','WMA100','HMA100','MACD','CCI','Stochastic Oscillator','RSI','ROC','PPO',
           'KST','BOLU','BOLD','BOLM']

for pair in os.listdir('DataReady'):
    if pair in ['EURUSD']:
        print(pair)
        data = pd.read_csv('DataReady/{}/{}_M15.csv'.format(pair, pair), names = columns, header = 0)
        #data = data[:-23000]
        toRemove = ['Volume', 'Date','High','Low','Open','Close']
        df = selectData(data,toRemove)
        closingPrices = data['Close']
        closingPrices = closingPrices.reset_index(drop=True)
        normDf = normalizeData(df)
        images = generateImages(normDf)
        images = np.array(images)

        train_X, _, train_Y, _ = train_test_split(images, closingPrices[28:], test_size = 0.2,shuffle = True, random_state = 42)
        _, test_X, _, test_Y = train_test_split(images, closingPrices[28:], test_size = 0.2,shuffle = False)
        train_X, test_X = train_X.reshape(train_X.shape[0],28*28).astype(np.float32), test_X.reshape(test_X.shape[0],28*28).astype(np.float32)
        
        for clf in classifiers.keys():
            print(clf)
            test_hat_Y = classifiers[clf].fit(train_X, train_Y).predict(test_X)
            best_param = 0
            try:
                best_param = classifiers[clf].best_params_
                print(best_param)
            except:
                print(0)
                
            mse = np.mean((test_hat_Y-test_Y)**2)
            corr = np.corrcoef(test_hat_Y, test_Y)[0,1]
            dti = _dtw(np.array(test_hat_Y), np.array(test_Y), 1)
            fast_dti = fastdtw(test_hat_Y, test_Y, 1)[0]
            print(fast_dti)
            
            with open("Results/ResultsML.txt","a+") as f:
                f.write("{}_M15,{},{},{},{},{},{}\n".format(pair, clf, mse, corr, dti, fast_dti, best_param))



from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Scripts.utils import *
import json
import pickle

classifiers = {
    "Linear Regression": LinearRegression(),
    "Bayesian Ridge Regression": BayesianRidge(),
    "Support Vector Regression": SVR(),
    "Random Forest Regression": RandomForestRegressor(),
    "KNN Regression": KNeighborsRegressor()
}

columns = ['Date','Open','High','Low','Close','Volume','Yield','PercentageVolume',
           'SMA6','EMA6','WMA6','HMA6','SMA20','EMA20','WMA20','HMA20','SMA50','EMA50','WMA50','HMA50',
           'SMA100','EMA100','WMA100','HMA100','MACD','CCI','Stochastic Oscillator','RSI','ROC','PPO',
           'KST','BOLU','BOLD','BOLM']

with open("Models/ML/MLModels.json","r") as f:
    models = json.load(f)
for pair in os.listdir('DataReady'):
    if len(pair)==6 and pair in ['EURUSD']:
        print(pair)
        data = pd.read_csv('DataReady/{}/{}_M15.csv'.format(pair, pair), names = columns, header = 0)
        toRemove = ['Volume', 'Date','High','Low','Open','Close']
        df = selectData(data,toRemove)
        closingPrices = data['Close']
        closingPrices = closingPrices.reset_index(drop=True)
        normDf = normalizeData(df)
        images = generateImages(normDf)
        images = np.array(images)

        train_X, _, train_Y, _ = train_test_split(images, closingPrices[28:], test_size = 0.2,shuffle = True, random_state = 42)
        train_X = train_X.reshape(train_X.shape[0],28*28).astype(np.float32)
        
        for clf in classifiers.keys():
            print(clf)
            model = classifiers[clf].set_params(**models[clf][pair])
            model.fit(train_X, train_Y)
            # save the model to disk
            filename = 'Models/{}_{}.sav'.format(clf.replace(" ",""),pair)
            pickle.dump(model, open(filename, 'wb'))
            #loaded_model = pickle.load(open(filename, 'rb'))
        


print('all done')
