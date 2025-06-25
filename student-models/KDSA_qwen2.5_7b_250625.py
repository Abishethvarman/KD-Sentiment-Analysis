import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import os

# Set parameters
model_id = "Qwen/Qwen2.5-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
input_csv = "../raw-dataset/movie.csv"
output_data_path = "../output-datasets/SA_qwen2.5_7b.csv"
results_csv_path = "../results/results.csv"

# Ensure output directories exist
os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

print(" Loading dataset...")
original_df = pd.read_csv(input_csv)

# Step 1: Load tokenizer and model
print(" Loading Qwen2.5-7B model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# Step 2: Define sentiment classification function
def classify_sentiment_qwen(text):
    prompt = f"Classify the sentiment of the following sentence as either Positive or Negative.\n\nSentence: \"{text}\"\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    if "positive" in decoded:
        return 1
    elif "negative" in decoded:
        return 0
    else:
        return -1

# Step 3: Run classification without modifying original df
print(" Predicting sentiment...")
texts = original_df["text"].tolist()
labels = original_df["label"].tolist()

predictions = []
for i, text in enumerate(texts):
    pred = classify_sentiment_qwen(text)
    predictions.append(pred)
    print(f"Processed {i+1}/{len(texts)}")

# Step 4: Create new dataframe for results
print(" Creating output DataFrame...")
results_df = pd.DataFrame({
    "text": texts,
    "label": labels,
    "predicted_label": predictions
})

# Filter valid predictions
results_df_clean = results_df[results_df["predicted_label"] != -1]

# Step 5: Evaluate
y_true = results_df_clean["label"]
y_pred = results_df_clean["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print("\n Classification Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix}")

# Step 6: Save predictions
print(f"\n Saving predictions to {output_data_path}")
results_df_clean.to_csv(output_data_path, index=False)

# Step 7: Save metrics to results.csv
print(f" Appending evaluation metrics to {results_csv_path}")
metrics_entry = pd.DataFrame([{
    "model": "Qwen2.5-7B",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "num_samples": len(results_df_clean)
}])

# Append to results file or create if it doesn't exist
if os.path.exists(results_csv_path):
    existing_results = pd.read_csv(results_csv_path)
    updated_results = pd.concat([existing_results, metrics_entry], ignore_index=True)
else:
    updated_results = metrics_entry

updated_results.to_csv(results_csv_path, index=False)
print(f" Results appended to: {results_csv_path}")
