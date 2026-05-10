import pandas as pd
import os
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
    text = text.lower()
    return text.strip()

def identify_reddit_columns(df):
    text_col = None
    label_col = None

    for col in df.columns:
        if "comment" in col.lower() or "body" in col.lower() or "text" in col.lower():
            text_col = col
        if "label" in col.lower() or "sarcasm" in col.lower():
            label_col = col

    if text_col is None:
        raise ValueError("Reddit verisinde yorum metni için uygun bir kolon bulunamadı!")

    if label_col is None:
        print("[WARNING] Reddit verisinde 'label' kolonu yok. Otomatik 0 ekleniyor.")
        df["label"] = 0
        label_col = "label"

    return text_col, label_col

def preprocess_datasets(
    headlines_path="data/processed/sarcasm_headlines.csv",
    reddit_path="data/raw/reddit_sarcasm.csv",
    save_path="data/processed/combined.csv"
):
    print("[INFO] Veriler yükleniyor...")
    df_head = pd.read_csv(headlines_path)
    df_red = pd.read_csv(reddit_path)

    print("[INFO] Temizleniyor ve düzenleniyor...")

    df_head = df_head[["headline", "is_sarcastic"]].rename(columns={"headline": "text", "is_sarcastic": "label"})
    df_head["text"] = df_head["text"].apply(clean_text)

    text_col, label_col = identify_reddit_columns(df_red)
    df_red = df_red[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df_red["text"] = df_red["text"].apply(clean_text)


    df_combined = pd.concat([df_head, df_red], ignore_index=True)
    df_combined.dropna(inplace=True)

    count_0 = df_combined[df_combined["label"] == 0]
    count_1 = df_combined[df_combined["label"] == 1]
    min_count = min(len(count_0), len(count_1))

    df_balanced = pd.concat([
        count_0.sample(min_count, random_state=42),
        count_1.sample(min_count, random_state=42)
    ], ignore_index=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_balanced.to_csv(save_path, index=False)
    print(f"[INFO] Kaydedildi: {save_path} | Toplam satır: {len(df_balanced)}")

    return df_balanced
