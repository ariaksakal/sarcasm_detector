# src/model.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

def train_model(
    data_path="data/processed/combined.csv",
    model_path="models/rf_model.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
):
    print("[INFO] Veri yükleniyor...")
    df = pd.read_csv(data_path)
    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"] != ""]

    X = df["text"]
    y = df["label"]

    print("[INFO] Veriler train/test olarak bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] TF-IDF vektörleri oluşturuluyor...")
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[INFO] Random Forest modeli eğitiliyor...")
    model = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    print("[INFO] Test verisi üzerinde değerlendiriliyor...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    print("[INFO] Model ve vektörizer kaydediliyor...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[SAVED] Model: {model_path}")
    print(f"[SAVED] Vektörizer: {vectorizer_path}")
