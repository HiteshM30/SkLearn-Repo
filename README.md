# SkLearn Repository

Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python.
 It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.
  This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.

# Importing Boston Housing CSV

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Using x,y variables to store data and target dataframe

x,y=data,target

# Importing Linear Regression Library

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Initializing a linear regression model

mod=LinearRegression()

# Training the model with x,y parameters

mod.fit(x,y)

# Making predictions

pred=mod.predict(x)
plt.scatter(pred,y)

# Importing the K-Neighbour Library

from sklearn.neighbors import KNeighborsRegressor

# Initializing a K-Neighbour Model

mod1= KNeighborsRegressor()

# Training the model

mod1.fit(x,y)

# Making the predictions

pred1=mod1.predict(x)
plt.scatter(pred1,y)

# PreProcessing

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Developing a pipeline to keep track and training the model as well

pipe=Pipeline([
    ("Pipe", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))])
pipe.fit(x,y)
pred=pipe.predict(x)
plt.scatter(pred,y)

# GridSearchCV Model
from sklearn.model_selection import GridSearchCV

# Function to get to know the parameters used by the KNeighbourRegressor

pipe.get_params()

# Making a GridSearch CV model with specific parameters, training it and predicting values from it

mod=GridSearchCV(
    estimator=pipe,
    param_grid={'model__n_neighbors':[1,2,3,4,5,6,7,8,9,10]}, 
    cv=3
)
mod.fit(x,y)
pred=mod.predict(x)
plt.scatter(pred,y)

# Analysing the results from the GridSearchCV Model

pd.DataFrame(mod.cv_results_)

