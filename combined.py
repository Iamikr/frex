from Scripts.utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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

# Load model parameters
with open("Models/ML/MLModels.json", "r") as f:
    params = json.load(f)

# Define models
models = {
    "Support Vector Regression": SVR(**remove_prefix_from_params(params["Support Vector Regression"]["EURUSD"])),
    "Linear Regression": LinearRegression(**remove_prefix_from_params(params["Linear Regression"]["EURUSD"])),
    "Bayesian Ridge Regression": BayesianRidge(**remove_prefix_from_params(params["Bayesian Ridge Regression"]["EURUSD"])),
    "KNN Regression": KNeighborsRegressor(**remove_prefix_from_params(params["KNN-Regression"]["EURUSD"])),
    "Random Forest Regression": RandomForestRegressor(**remove_prefix_from_params(params["Random Forest Regression"]["EURUSD"])),
    "Gradient Boosting Regression": GradientBoostingRegressor(**remove_prefix_from_params(params["Gradient Boosting Regression"]["EURUSD"])),
    "AdaBoost Regression": AdaBoostRegressor(**remove_prefix_from_params(params["AdaBoost Regression"]["EURUSD"])),
    "ElasticNet Regression": ElasticNet(**remove_prefix_from_params(params["ElasticNet Regression"]["EURUSD"])),
    "Lasso Regression": Lasso(**remove_prefix_from_params(params["Lasso Regression"]["EURUSD"])),
    "Ridge Regression": Ridge(**remove_prefix_from_params(params["Ridge Regression"]["EURUSD"])),
    "XGBoost Regression": XGBRegressor(**remove_prefix_from_params(params["XGBoost Regression"]["EURUSD"])),
    "LightGBM Regression": LGBMRegressor(**remove_prefix_from_params(params["LightGBM Regression"]["EURUSD"]))
}

# Predictions for each model
predictions = {}
for model_name, model in models.items():
    print(f"Training {model_name}")
    model.fit(train_X, train_Y)
    predictedPrices = model.predict(test_X)
    predictions[model_name] = predictedPrices
    print(f"MSE: {np.mean((predictedPrices - test_Y) ** 2):e}")
    print(f"CORR: {np.corrcoef(predictedPrices, test_Y)[0, 1]:.3f}")
    print(f"DTW: {DTW(predictedPrices, np.array(test_Y), 1):.3f}")
    print(f"FastDTW: {fastdtw(predictedPrices, np.array(test_Y), 1)[0]:.3f}")

    plt.figure(figsize=(30, 10))
    plt.grid()
    plt.plot(range(len(predictedPrices)), predictedPrices, color='darkorange', marker='^', label=model_name)
    plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
    plt.legend(fontsize=20)
    plt.show()

# Combined predictions by averaging
combined_predictions = np.mean(list(predictions.values()), axis=0)
print(f"Combined MSE: {np.mean((combined_predictions - test_Y) ** 2):e}")
print(f"Combined CORR: {np.corrcoef(combined_predictions, test_Y)[0, 1]:.3f}")
print(f"Combined DTW: {DTW(combined_predictions, np.array(test_Y), 1):.3f}")
print(f"Combined FastDTW: {fastdtw(combined_predictions, np.array(test_Y), 1)[0]:.3f}")

plt.figure(figsize=(30, 10))
plt.grid()
plt.plot(range(len(combined_predictions)), combined_predictions, color='darkorange', marker='^', label='Combined Predictions')
plt.plot(range(len(test_Y)), test_Y, color='blue', marker='o', label='true')
plt.legend(fontsize=20)
plt.show()
