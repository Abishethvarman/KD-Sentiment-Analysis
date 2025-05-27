import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset
from sklearn.metrics import classification_report
import os
from tqdm import tqdm

# Parameters
teacher_model_id = "meta-llama/Llama-3.1-70B"  # Change this to a bigger model like roberta-large if needed
student_model_id = "meta-llama/Llama-3.1-8B"   # Small student model
save_path = "./saved_models/llama3.1_8b_distill_llama3.1_70b"
csv_path = "movie.csv"
num_labels = 2
batch_size = 16
num_epochs = 3
learning_rate = 2e-5
temperature = 2.0
alpha = 0.7  # weight for distillation loss vs. hard label loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv(csv_path)
df = df[["text", "label"]].dropna()
dataset = Dataset.from_pandas(df)

# Load tokenizers and models
tokenizer = AutoTokenizer.from_pretrained(student_model_id)

teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_id, num_labels=num_labels).to(device)
teacher_model.eval()

student_model = AutoModelForSequenceClassification.from_pretrained(student_model_id, num_labels=num_labels).to(device)

# Tokenization
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader
train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

# Loss functions
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
ce_loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = AdamW(student_model.parameters(), lr=learning_rate)

# Distillation training loop
print("🔁 Starting distillation training...")

for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Teacher outputs
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
            soft_labels = torch.softmax(teacher_logits / temperature, dim=1)

        # Student outputs
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        soft_student = torch.log_softmax(student_logits / temperature, dim=1)

        # Compute losses
        loss_kl = kl_loss_fn(soft_student, soft_labels) * (temperature ** 2)
        loss_ce = ce_loss_fn(student_logits, labels)
        loss = alpha * loss_kl + (1 - alpha) * loss_ce

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

# Save the distilled model
os.makedirs(save_path, exist_ok=True)
student_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n💾 Distilled student model saved to: {save_path}")

# Evaluation
print("\n📊 Evaluating the distilled student model...")
student_model.eval()

def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = student_model(**inputs).logits
    return torch.argmax(outputs, dim=1).cpu().numpy()

preds = []
labels = df["label"].tolist()
for i in range(0, len(df), batch_size):
    batch_texts = df["text"].iloc[i:i+batch_size].tolist()
    preds.extend(predict(batch_texts))

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
