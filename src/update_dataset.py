import pandas as pd
from src.preprocess import preprocess_datasets

def load_feedback_data(path="feedback_log.csv"):
    df = pd.read_csv(path, names=["text", "label_str", "confidence", "feedback"])
    df = df[df["feedback"] == "no"]  # Sadece yanlış tahminleri al
    df = df[df["label_str"].isin(["Sarcastic", "Not Sarcastic"])]

    df["label"] = df["label_str"].map({"Sarcastic": 1, "Not Sarcastic": 0})
    return df[["text", "label"]]

def get_combined_dataset():
    original_df = preprocess_datasets()
    feedback_df = load_feedback_data()
    combined = pd.concat([original_df, feedback_df], ignore_index=True)
    return combined

if __name__ == "__main__":
    df = get_combined_dataset()
    print(f"Birleşik veri boyutu: {df.shape}")
    print(df.sample(5))
    