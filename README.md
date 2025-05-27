# KD-Sentiment-Analysis


## 📁 Project Structure

```text
KD-Sentiment-Analysis/
├── distillation/               # Code related to the distillation process
├── distilled-models-saved/     # Stores distilled (student) models after training
├── raw-datasets/               # Original input datasets 
├── student-models/             # results for student models       # Summarized versions of the datasets
├── teacher-models/             # evaluation results for teacher models
├── venv/                       # Python virtual environment (excluded in .gitignore)
├── .gitignore
├── README.md
└── requirements.txt
```


## 🏃 Execution Order

Please run the files in the following order:

### LLaMA 3.1 Models
- `KDSA_llama3.1_8b_270525.py`
- `KDSA_llama3.1_70b_270525.py`
- `KDSA_llama3.1_8b_distill_llama3.1_70b_270525.py`

### HuggingFace Export
- `Huggingface_exporter.py`  
  ⚠️ Make sure to configure paths, repo name, and HuggingFace credentials before running this script.



## ✅ Requirements

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
