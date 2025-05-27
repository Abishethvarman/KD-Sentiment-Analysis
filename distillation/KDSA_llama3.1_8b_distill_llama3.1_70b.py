import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
import os

# Parameters
teacher_model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
student_model_id = "google/gemma-2b"  # or another small LLM like "bert-base-uncased"
save_path = "./saved_models/llama_distilled"
csv_path = "movie.csv"

# Step 1: Load dataset
df = pd.read_csv(csv_path)
df = df[["text", "label"]].dropna()

# Step 2: Create Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 3: Load teacher model (causal LM)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_id, device_map="auto", torch_dtype=torch.float16)

# Soft-label inference function
def get_teacher_soft_label(example):
    prompt = f"Classify the sentiment of the following sentence as either Positive or Negative.\n\nSentence: \"{example['text']}\"\nSentiment:"
    inputs = teacher_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = teacher_model.generate(**inputs, max_new_tokens=10)
    decoded = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    example["teacher_label"] = 1 if "positive" in decoded else 0
    return example

print("Generating soft labels using teacher model...")
dataset = dataset.map(get_teacher_soft_label)

# Step 4: Load student model (sequence classification)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_id, num_labels=2)

# Tokenize for student
def tokenize_function(example):
    return student_tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Rename teacher_label to labels
tokenized_dataset = tokenized_dataset.rename_column("teacher_label", "labels")
tokenized_dataset.set_format("torch")

# Step 5: Train student model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # optional, same for demo
    tokenizer=student_tokenizer,
)

print("Training student model...")
trainer.train()

# Step 6: Save student model
os.makedirs(save_path, exist_ok=True)
student_model.save_pretrained(save_path)
student_tokenizer.save_pretrained(save_path)
print(f"\n✅ Distilled student model saved to: {save_path}")

# Step 7: Evaluate distilled model
def predict_sentiment(texts):
    inputs = student_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = student_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy()

print("\nEvaluating distilled model on actual labels...")
preds = predict_sentiment(df["text"].tolist())

print("\nClassification Report:")
print(classification_report(df["label"], preds, target_names=["Negative", "Positive"]))
