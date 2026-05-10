import torch
from torch.utils.data import Dataset
import pandas as pd
from src.preprocess import preprocess_datasets

class SarcasmDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]["text"])
        label = int(self.data.iloc[index]["label"])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

def get_combined_data():
    df = preprocess_datasets()
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    return df
