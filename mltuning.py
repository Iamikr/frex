import json
import pickle
from Scripts.utils import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Scripts.FastDTW import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from tqdm import tqdm

def _dtw(output, target, window):
    n, m = len(output), len(target)
    w = np.max([window, abs(n - m)])
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix += float("Inf")
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        a, b = np.max([1, i - w]), np.min([m, i + w]) + 1
        dtw_matrix[i, a:b] = 0
        for j in range(a, b):
            cost = np.abs(output[i - 1] - target[j - 1])
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1, -1]

params_svr = {
    "kernel": ['rbf', 'linear', 'poly'],
    "C": [0.01, 0.1, 1, 10],
    "gamma": [0.0001, 0.001, 0.01],
    "degree": [1, 2, 3]
}
gssvr = GridSearchCV(SVR(), params_svr, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

params_rfg = {
    "n_estimators": [50, 100, 500]
}
gsrfg = GridSearchCV(RandomForestRegressor(), params_rfg, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

params_br = {
    "max_iter": [100, 1000, 2000],
    "tol": [1e-6, 1e-8, 1e-10]
}
gsbr = GridSearchCV(BayesianRidge(), params_br, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

params_knn = {
    "n_neighbors": [50, 100, 300, 1000],
    "weights": ['uniform', 'distance'],
    "algorithm": ['ball_tree', 'kd_tree', 'brute']
}
gsknn = GridSearchCV(KNeighborsRegressor(), params_knn, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

classifiers = {
    "Linear Regression": LinearRegression(),
    "Bayesian Ridge Regression": gsbr,
    "Support Vector Regression": gssvr,
    "Random Forest Regression": gsrfg,
    "KNN-Regression": gsknn
}

columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Yield', 'PercentageVolume',
           'SMA6', 'EMA6', 'WMA6', 'HMA6', 'SMA20', 'EMA20', 'WMA20', 'HMA20', 'SMA50', 'EMA50', 'WMA50', 'HMA50',
           'SMA100', 'EMA100', 'WMA100', 'HMA100', 'MACD', 'CCI', 'Stochastic Oscillator', 'RSI', 'ROC', 'PPO',
           'KST', 'BOLU', 'BOLD', 'BOLM']

with open("Models/ML/MLModels.json", "r") as f:
    models = json.load(f)

for pair in os.listdir('DataReady'):
    if pair in ['EURUSD', 'EURGBP', 'EURCAD', 'EURAUD', 'EURJPY', 'EURCHF', 'USDJPY', 'USDCAD', 'AUDCAD', 'GBPUSD', 'AUDUSD']:
        print(pair)
        data = pd.read_csv(f'DataReady/{pair}/{pair}_H4.csv', names=columns, header=0)
        toRemove = ['Volume', 'Date', 'High', 'Low', 'Open', 'Close']
        df = selectData(data, toRemove)
        closingPrices = data['Close']
        closingPrices = closingPrices.reset_index(drop=True)
        normDf = normalizeData(df)
        images = generateImages(normDf)
        images = np.array(images)

        train_X, _, train_Y, _ = train_test_split(images, closingPrices[28:], test_size=0.2, shuffle=True, random_state=42)
        train_X = train_X.reshape(train_X.shape[0], 28 * 28).astype(np.float32)

        for clf in tqdm(classifiers.keys(), desc="Training classifiers"):
            print(clf, 'is training')
            if clf in models and pair in models[clf]:
                if isinstance(classifiers[clf], GridSearchCV):
                    classifiers[clf].estimator.set_params(**models[clf][pair])
                else:
                    classifiers[clf].set_params(**models[clf][pair])
            classifiers[clf].fit(train_X, train_Y)
            print('before saving')
            filename = f'Models/{clf.replace(" ", "")}_{pair}.sav'
            print(filename, 'saved')
            pickle.dump(classifiers[clf], open(filename, 'wb'))
