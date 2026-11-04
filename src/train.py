# === Importing Libraries ==============================

import os

import time

import pandas as pd

from feature_engineering import feature_construct

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import joblib


# === Importing Dataset ==============================

data = pd.read_csv("data/email.csv")


# === Data Cleaning ==============================

data.drop_duplicates(keep="first", inplace=True)


# === Splitting Data ==============================

X = data.drop(columns=['Category'])
y = data['Category']



# === Building Model ==============================

model = Pipeline(steps=[
    ('feature_construct', FunctionTransformer(feature_construct)),
    ('preprocessing', StandardScaler()),
    ('model', LogisticRegression(C=10, class_weight=None))
])

start_time = time.time()
model.fit(X,y)
end_time = time.time()- start_time

print("Model Trained Successfully!")
print(f"Time Taken for Training: {end_time:.2f}s\n" )


# === Saving Model ==============================

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/SpamLens.pkl')
print("Model Saved Successfully!")