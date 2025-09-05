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
    df = df.rename(columns=col_map)

    # 🔧 Ensure Outcome is always numeric  # <<< ADD THIS
    if 'Outcome' in df.columns:           # <<< ADD THIS
        if df['Outcome'].dtype == object: # <<< ADD THIS
            df['Outcome'] = (             # <<< ADD THIS
                df['Outcome']             # <<< ADD THIS
                .astype(str)              # <<< ADD THIS
                .str.strip()              # <<< ADD THIS
                .str.lower()              # <<< ADD THIS
                .replace({'yes': 1, 'no': 0, 'positive': 1, 'negative': 0,   # <<< ADD THIS
                          'tested_positive': 1, 'tested_negative': 0})       # <<< ADD THIS
            )                             # <<< ADD THIS

        try:                              # <<< ADD THIS
            df['Outcome'] = df['Outcome'].astype(int)  # <<< ADD THIS
        except ValueError:                # <<< ADD THIS
            raise ValueError(f"Cannot convert Outcome column to integers. Found values: {df['Outcome'].unique()}")  # <<< ADD THIS

    return df


def get_feature_target(df: pd.DataFrame):
    """
    Extracts features (X) and target (y) from the dataset.
    Automatically detects target column and converts to numeric.
    Works with 0/1, Yes/No, Positive/Negative, Diabetic/Non-diabetic.
    """

    # 1. Candidate target columns (auto-detect)
    possible_targets = ['Outcome', 'Diabetes', 'Class', 'Target']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"No valid target column found! Expected one of: {possible_targets}")

    # 2. Extract target
    y = df[target_col].copy()

    # 3. Convert target to numeric (0/1)
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower()
        mapping = {
            'yes': 1, 'no': 0,
            'positive': 1, 'negative': 0,
            'diabetic': 1, 'nondiabetic': 0
        }
        y = y.replace(mapping)

    try:
        y = y.astype(int)
    except ValueError:
        raise ValueError(f"Cannot convert target column '{target_col}' to integers. Found values: {y.unique()}")

    # 4. Features = everything except target column
    X = df.drop(columns=[target_col]).copy()

    return X, y


def train_test(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    X, y = get_feature_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
