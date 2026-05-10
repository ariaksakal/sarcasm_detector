from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
from src.utils import SarcasmDataset, get_combined_data
import os

def train_bert_model(model_save_path="models/bert_model.pt", epochs=10, batch_size=16, learning_rate=3e-5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    df = get_combined_data()
    dataset = SarcasmDataset(df, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    loss_fn = CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch in tqdm(dataloader, desc="🔄 Eğitim"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"📉 Ortalama Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Eğitim tamamlandı. Model kaydedildi: {model_save_path}")
