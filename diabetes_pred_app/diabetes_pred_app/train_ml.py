from __future__ import annotations
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import save_joblib, ensure_dir
from data_loader import fetch_and_cache_dataset, train_test

MODELS_DIR = 'models'

def train_ml_models():
    df = fetch_and_cache_dataset()
    X_train, X_test, y_train, y_test = train_test(df, test_size=0.2)

    # Random Forest (no scaling needed)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, (rf_proba >= 0.5).astype(int))
    rf_auc = roc_auc_score(y_test, rf_proba)
    ensure_dir(MODELS_DIR)
    save_joblib(rf, os.path.join(MODELS_DIR, 'random_forest.joblib'))

    # Logistic Regression (with scaling)
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, (lr_proba >= 0.5).astype(int))
    lr_auc = roc_auc_score(y_test, lr_proba)
    save_joblib(lr, os.path.join(MODELS_DIR, 'log_reg_scaled.joblib'))

    metrics = {
        'random_forest': {'accuracy': float(rf_acc), 'roc_auc': float(rf_auc)},
        'log_reg_scaled': {'accuracy': float(lr_acc), 'roc_auc': float(lr_auc)}
    }
    return metrics

if __name__ == '__main__':
    m = train_ml_models()
    print(m)
