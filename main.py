from src.load_data import download_headlines_dataset, download_reddit_dataset
from src.preprocess import preprocess_datasets
from src.bert_model import train_bert_model
import os

if __name__ == "__main__":
    print("🧪 [1/3] Headlines verisi indiriliyor...")
    df_head = download_headlines_dataset()
    print(f"✅ Headlines boyutu: {df_head.shape}")

    print("\n🧪 [2/3] Reddit verisi indiriliyor...")
    df_red = download_reddit_dataset()
    print(f"✅ Reddit boyutu: {df_red.shape}")

    print("\n🧼 [3/3] Veriler temizleniyor ve birleştiriliyor...")
    df_all = preprocess_datasets()
    print(f"✅ Birleşik veri boyutu: {df_all.shape}")

    print("\n🔥 [EĞİTİM] BERT modeli eğitiliyor (15 epoch, batch size 32)...")
    train_bert_model(df_all, epochs=20, batch_size=32)
    print("✅ Eğitim tamamlandı. Model 'models/bert_model.pt' olarak kaydedildi.")