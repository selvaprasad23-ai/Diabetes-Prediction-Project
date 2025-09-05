from __future__ import annotations
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow import keras
from utils import ensure_dir
from data_loader import fetch_and_cache_dataset, train_test

MODELS_DIR = 'models'
DL_MODEL_PATH = os.path.join(MODELS_DIR, 'keras_dense.keras')
RANDOM_STATE = 42

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

def train_dl_model(epochs: int = 40, batch_size: int = 32):
    df = fetch_and_cache_dataset()
    X_train, X_test, y_train, y_test = train_test(df, test_size=0.2)

    # Scale inputs for DL
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Persist the scaler for use at inference time
    import joblib
    ensure_dir(MODELS_DIR)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'dl_scaler.joblib'))

    model = build_model(X_train_s.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_auc', mode='max')
    ]
    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate
    proba = model.predict(X_test_s, verbose=0).ravel()
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y_test, proba)

    # Save
    ensure_dir(MODELS_DIR)
    model.save(DL_MODEL_PATH)

    return {'dl_dense': {'accuracy': float(acc), 'roc_auc': float(auc)}}

if __name__ == '__main__':
    m = train_dl_model()
    print(m)
