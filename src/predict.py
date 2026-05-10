import joblib
import os

def predict_sarcasm(text, model_path="models/rf_model.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model ya da TF-IDF vektörizer dosyası bulunamadı!")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    text_cleaned = text.lower().strip()
    vec = vectorizer.transform([text_cleaned])
    pred = model.predict(vec)[0]


    prob = model.predict_proba(vec)[0][pred] if hasattr(model, "predict_proba") else 1.0

    return {
        "text": text,
        "is_sarcastic": bool(pred),
        "confidence": round(prob * 100, 2)
    }
