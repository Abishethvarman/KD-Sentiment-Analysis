import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AdamW
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tqdm import tqdm

# === Parameters ===
teacher_model_id = "meta-llama/Llama-3.1-70B"
student_model_id = "meta-llama/Llama-3.1-8B"
csv_path = "../raw-dataset/movie.csv"
save_model_path = "../saved_models/llama3.1_8b_distill_llama3.1_70b"
output_csv_path = "../output-datasets/SA_llama3.1_8b_distill_llama3.1_70b.csv"
results_csv_path = "../results/results.csv"

num_labels = 2
batch_size = 16
num_epochs = 3
learning_rate = 2e-5
temperature = 2.0
alpha = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Ensure directories exist ===
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(csv_path).dropna()
df = df[["text", "label"]]
dataset = Dataset.from_pandas(df)

# === Load tokenizer and models ===
print("🔁 Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(student_model_id)
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_id, num_labels=num_labels).to(device)
teacher_model.eval()
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_id, num_labels=num_labels).to(device)

# === Tokenization ===
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

# === Loss & Optimizer ===
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
ce_loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(student_model.parameters(), lr=learning_rate)

# === Distillation Training ===
print(" Starting distillation training...")
for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
            soft_labels = torch.softmax(teacher_logits / temperature, dim=1)

        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        soft_student = torch.log_softmax(student_logits / temperature, dim=1)

        loss_kl = kl_loss_fn(soft_student, soft_labels) * (temperature ** 2)
        loss_ce = ce_loss_fn(student_logits, labels)
        loss = alpha * loss_kl + (1 - alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

# === Save distilled model ===
print(f"\n💾 Saving distilled student model to: {save_model_path}")
student_model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)

# === Evaluation: Prompt-based inference ===
print("\n📊 Evaluating the distilled model..")

# Use causal model interface for prompts
eval_model = AutoModelForCausalLM.from_pretrained(save_model_path, device_map="auto", torch_dtype=torch.float16)
eval_tokenizer = AutoTokenizer.from_pretrained(save_model_path)

def classify_prompt(text):
    prompt = f"Classify the sentiment of the following sentence as either Positive or Negative.\n\nSentence: \"{text}\"\nSentiment:"
    inputs = eval_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = eval_model.generate(**inputs, max_new_tokens=10)
    decoded = eval_tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    if "positive" in decoded:
        return 1
    elif "negative" in decoded:
        return 0
    else:
        return -1

predictions = []
texts = df["text"].tolist()
labels = df["label"].tolist()

for i, text in enumerate(texts):
    pred = classify_prompt(text)
    predictions.append(pred)
    print(f"Processed {i+1}/{len(texts)}")

# === Create output DataFrame ===
results_df = pd.DataFrame({
    "text": texts,
    "label": labels,
    "predicted_label": predictions
})
results_df_clean = results_df[results_df["predicted_label"] != -1]

# === Metrics ===
y_true = results_df_clean["label"]
y_pred = results_df_clean["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print("\n Final Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# === Save output predictions CSV ===
results_df_clean.to_csv(output_csv_path, index=False)
print(f"\n📁 Output predictions saved to: {output_csv_path}")

# === Append evaluation to results.csv ===
metrics_entry = pd.DataFrame([{
    "model": "llama3.1_8b_distilled_from_70b",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "num_samples": len(results_df_clean)
}])

if os.path.exists(results_csv_path):
    existing = pd.read_csv(results_csv_path)
    updated = pd.concat([existing, metrics_entry], ignore_index=True)
else:
    updated = metrics_entry

updated.to_csv(results_csv_path, index=False)
print(f"📊 Evaluation results appended to: {results_csv_path}")
