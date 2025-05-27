import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# Step 1: Load your dataset
df = pd.read_csv("movie.csv")

# Step 2: Load LLaMA3.1-70B-Instruct model
model_id = "meta-llama/Llama-3.1-70B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",                # distribute across GPUs if available
    torch_dtype=torch.float16         # use float16 to save memory
)

# Step 3: Define sentiment classification function using prompts
def classify_sentiment_llama(text):
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

# Step 4: Apply the model to predict sentiment
print("Classifying each row using LLaMA 3.1 70B...")
df["llama3.1_70b"] = df["text"].apply(classify_sentiment_llama)

# Step 5: Filter out rows with unknown predictions
df_clean = df[df["llama3.1_70b"] != -1]

# Step 6: Evaluate model performance
print("\nClassification Report:")
print(classification_report(df_clean["label"], df_clean["llama3.1_70b"], target_names=["Negative", "Positive"]))

print("\nConfusion Matrix:")
print(confusion_matrix(df_clean["label"], df_clean["llama3.1_70b"]))

print("\nAccuracy:")
print(accuracy_score(df_clean["label"], df_clean["llama3.1_70b"]))

# Step 7: Save final CSV
output_path = "../datasets-classified/llama3.1_70b.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df_clean[["text", "label", "llama3.1_70b"]].to_csv(output_path, index=False)
print(f"\n✅ Saved the results to: {output_path}")
