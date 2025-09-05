import os
import joblib

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_joblib(obj, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)

def load_joblib(path: str):
    return joblib.load(path)
