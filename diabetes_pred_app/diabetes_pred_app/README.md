# Diabetes Prediction App (ML + DL)

A ready-to-run Streamlit app that predicts diabetes using **Machine Learning (Random Forest + Logistic Regression)** and **Deep Learning (Keras Dense NN)**.

## 🚀 Quick Start

1. **Create & activate a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

The app will download the **Pima Indians Diabetes** dataset from OpenML on first run and cache it under `data/diabetes_openml.csv`. Models are saved under `models/`.

## 📂 Project Structure

```
diabetes_pred_app/
├── app.py                 # Streamlit UI
├── data_loader.py         # Dataset fetch + clean + split
├── train_ml.py            # Trains ML models (RF, LogisticRegression)
├── train_dl.py            # Trains Keras Deep Learning model
├── utils.py               # Shared helpers
├── models/                # Saved models (.joblib, .keras)
├── data/                  # Cached dataset CSV
├── requirements.txt
└── README.md
```

## 🧠 Features

- Compare **ML vs DL** predictions side-by-side
- Probability scores with threshold control
- Training-from-UI with progress
- Sensible preprocessing (train/test split, scaling for DL, class balance check)
- Reproducible (fixed random_state)

## 📊 Inputs (Pima Diabetes)

- Pregnancies (0–17)
- Glucose (0–200)
- BloodPressure (0–140)
- SkinThickness (0–100)
- Insulin (0–900)
- BMI (0–70)
- DiabetesPedigreeFunction (0.0–2.5)
- Age (21–100)

## 🛠 Troubleshooting

- **TensorFlow install issues**: If TensorFlow is heavy for your machine, you can still use ML models. Comment out DL sections in `app.py` and omit TF from `requirements.txt`.
- **No Internet**: Place a CSV named `diabetes_openml.csv` in the `data/` folder with the Pima dataset (columns: pregnancies, plas, pres, skin, insu, mass, pedi, age, class).
- **GPU** is not required.

## 📜 License

MIT
