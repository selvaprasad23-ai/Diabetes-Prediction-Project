from __future__ import annotations
import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from typing import Tuple

CACHE_PATH = os.path.join('data', 'diabetes_openml.csv')
RANDOM_STATE = 42

# OpenML dataset 'diabetes' (Pima Indians Diabetes)
# Features in OpenML: pregnancies, plas, pres, skin, insu, mass, pedi, age
# Target: class (tested_positive/tested_negative)

def fetch_and_cache_dataset() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    df = fetch_openml(name='diabetes', version=1, as_frame=True).frame
    # Normalize column names to friendlier ones
    col_map = {
        'preg': 'Pregnancies',
        'plas': 'Glucose',
        'pres': 'BloodPressure',
        'skin': 'SkinThickness',
        'insu': 'Insulin',
        'mass': 'BMI',
        'pedi': 'DiabetesPedigreeFunction',
        'age': 'Age',
        'class': 'Outcome'
    }
    # Some versions use slightly different short names; handle both
    df = df.rename(columns={
        'preg': 'Pregnancies',
        'plas': 'Glucose',
        'pres': 'BloodPressure',
        'skin': 'SkinThickness',
        'insu': 'Insulin',
        'mass': 'BMI',
        'pedi': 'DiabetesPedigreeFunction',
        'age': 'Age',
        'class': 'Outcome'
    })
    # If columns already have long names (Kaggle-style), keep them
    if 'Outcome' not in df.columns and 'class' in df.columns:
        df = df.rename(columns={'class': 'Outcome'})
    # Convert outcome to 0/1
    if df['Outcome'].dtype == object:
        df['Outcome'] = (df['Outcome'].str.contains('positive', case=False)).astype(int)
    # Save cache
    os.makedirs('data', exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df

def get_feature_target(df: pd.DataFrame):
    feature_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    X = df[feature_cols].copy()
    y = df['Outcome'].astype(int).copy()
    return X, y

def train_test(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    X, y = get_feature_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
