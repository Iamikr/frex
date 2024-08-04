from Scripts.utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
import json
import pickle
from Scripts.DTW import *
from Scripts.FastDTW import *

# Helper function to remove 'estimator__' prefix
def remove_prefix_from_params(params, prefix='estimator__'):
    return {key.replace(prefix, ''): value for key, value in params.items() if key.startswith(prefix)}

# Load data and preprocess
columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Yield', 'PercentageVolume',
           'SMA6', 'EMA6', 'WMA6', 'HMA6', 'SMA20', 'EMA20', 'WMA20', 'HMA20', 'SMA50', 'EMA50', 'WMA50', 'HMA50',
           'SMA100', 'EMA100', 'WMA100', 'HMA100', 'MACD', 'CCI', 'Stochastic Oscillator', 'RSI', 'ROC', 'PPO',
           'KST', 'BOLU', 'BOLD', 'BOLM']
data = pd.read_csv('DataReady/EURUSD/EURUSD_H4.csv', names=columns, header=0)
toRemove = ['Volume', 'Date', 'High', 'Low', 'Open', 'Close']
df = selectData(data, toRemove)
close = data['Close']

# Normalize data
normDf = normalizeData(df)
normClose = normalizeData(close)

# Feature selection: Select top 10 features
selector = SelectKBest(score_func=f_regression, k=10)
selected_features = selector.fit_transform(normDf, normClose)
selected_feature_names = normDf.columns[selector.get_support()]

# Generate images (assuming this function is based on selected features)
images = generateImages(pd.DataFrame(selected_features, columns=selected_feature_names))
images = np.array(images)

# Ensure consistent length
min_len = min(len(images), len(close) - 28)
images = images[:min_len]
close = close[28:28 + min_len]

# Training and test datasets
train_X, _, train_Y, _ = train_test_split(images, close, test_size=0.2, shuffle=True, random_state=42)
_, test_X, _, test_Y = train_test_split(images, close, test_size=0.2, shuffle=False)
train_X = train_X.reshape(train_X.shape[0], -1).astype(np.float32)
test_X = test_X.reshape(test_X.shape[0], -1).astype(np.float32)

# SVR
with open("Models/ML/MLModels.json", "r") as f:
    params = json.load(f)["Support Vector Regression"]["EURUSD"]
params = remove_prefix_from_params(params)
svr = SVR(**params)
predictedPrices = svr.fit(train_X, train_Y).predict(test_X)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='SVR')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()

# Linear Regression
with open("Models/ML/MLModels.json", "r") as f:
    params = json.load(f)["Linear Regression"]["EURUSD"]
lr = LinearRegression(**params)
predictedPrices = lr.fit(train_X, train_Y).predict(test_X)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='Linear Regression')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()

# Polynomial Features (degree 2)
transform = PolynomialFeatures(2)
train_X2, test_X2 = transform.fit_transform(train_X), transform.fit_transform(test_X)
predictedPrices = lr.fit(train_X2, train_Y).predict(test_X2)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='Linear Regression - Polynomial Features (Degree 2)')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()

# Polynomial Features (degree 3)
transform = PolynomialFeatures(3)
train_X3, test_X3 = transform.fit_transform(train_X), transform.fit_transform(test_X)
predictedPrices = lr.fit(train_X3, train_Y).predict(test_X3)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='Linear Regression - Polynomial Features (Degree 3)')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()

# Random Forest Regression
with open("Models/ML/MLModels.json", "r") as f:
    params = json.load(f)["Random Forest Regression"]["EURUSD"]
params = remove_prefix_from_params(params)
rfg = RandomForestRegressor(**params)
predictedPrices = rfg.fit(train_X, train_Y).predict(test_X)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='Random Forest Regression')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()

features_importances = rfg.feature_importances_
impurityFeatures = [(name, sum(features_importances[i::10])) for name, i in zip(selected_feature_names, range(10))]
impurityFeatures.sort(key=lambda x: x[1], reverse=True)
print("Explained variance by the features:\n")
print("\n".join([elem[0] + " " + str(elem[1]) for elem in impurityFeatures]))

# Bayesian Ridge Regression
with open("Models/ML/MLModels.json", "r") as f:
    params = json.load(f)["Bayesian Ridge Regression"]["EURUSD"]
params = remove_prefix_from_params(params)
brr = BayesianRidge(**params)
predictedPrices = brr.fit(train_X, train_Y).predict(test_X)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='Bayesian Ridge Regression')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()

# KNN Regression
with open("Models/ML/MLModels.json", "r") as f:
    params = json.load(f)["KNN-Regression"]["EURUSD"]
params = remove_prefix_from_params(params)
knn = KNeighborsRegressor(**params)
predictedPrices = knn.fit(train_X, train_Y).predict(test_X)
print("MSE: {:e}".format(np.mean((predictedPrices - test_Y) ** 2)))
print("CORR: {:.3f}".format(np.corrcoef(predictedPrices, test_Y)[0, 1]))
print("DTW: {:.3f}".format(DTW(predictedPrices, np.array(test_Y), 1)))
print("DTW: {:.3f}".format(fastdtw(predictedPrices, np.array(test_Y), 1)[0]))

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label='KNN Regression')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()
