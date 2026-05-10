# src/load_data.py
from datasets import load_dataset
import pandas as pd
import os

def download_headlines_dataset(save_path="data/processed/sarcasm_headlines.csv"):
    print("[INFO] Loading sarcasm headlines dataset from local file...")

    if os.path.exists("data/processed") and not os.path.isdir("data/processed"):
        os.remove("data/processed")
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_json("data/raw/Sarcasm_Headlines_Dataset.json", lines=True)
    df.to_csv(save_path, index=False)
    return df

def download_reddit_dataset(save_path="data/raw/reddit_sarcasm.csv"):
    print("[INFO] Loading Reddit sarcasm dataset from HuggingFace...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ds = load_dataset("Thewillonline/reddit-sarcasm", split="train")
    df = pd.DataFrame(ds)

   
    df = df.sample(50000, random_state=42).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved Reddit data to {save_path} (rows: {len(df)})")
    return df
