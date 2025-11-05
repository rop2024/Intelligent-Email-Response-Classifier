
# Intelligent Email Reply Classifier

Short project description
-------------------------

This repository contains a small service and training pipeline for classifying email replies (or short text messages) into reply classes used by an intelligent reply system. It provides:

- A FastAPI app (`app.py`) exposing a /predict endpoint for single-text inference.
- Training utilities (`train.py`) to fine-tune a Hugging Face transformer on your labeled dataset.
- A lightweight baseline model (scikit-learn pipeline) and an HF model under `models/`.

How to run locally
------------------

1) Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2) Ensure models are present under `models/`:

- HF model path: `models/hf_model/` (contains tokenizer and model files)
- Baseline model: `models/baseline_model.pkl`
- Label mapping: `models/id2label.json`

3) Run the API with uvicorn (from project root):

```powershell
# from project root
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Note: `app.py` contains a top-level flag `USE_HF = True`. Set to `False` to use the baseline scikit-learn pipeline if you prefer the CPU/fast option.

Example curl requests
---------------------

Predict a single text (JSON POST):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text":"Could you please reschedule our meeting to next Tuesday?"}'
```

Example response:

```json
{
  "label": "reschedule",
  "confidence": 0.987
}
```

How the models were trained
---------------------------

- The training pipeline is implemented in `train.py`. There's also an exploratory notebook at `notebook.ipynb` used for experiments and quick checks.
- Typical training command (from project root):

```powershell
python train.py --data-file path/to/data.csv --model-name distilbert-base-uncased --output-dir models/hf_model --num-train-epochs 3
```

Options you may want:
- `--fp16` to enable mixed precision (faster on compatible GPUs)
- `--per-device-train-batch-size` to tune memory vs throughput

Which model is chosen for production and why
-------------------------------------------

The repository contains a `models/model_selection.json` file with the selection decision. Current choice:

- selected_model: `baseline`
- baseline_accuracy: ~0.99765
- hf_accuracy: 1.0
- reason: "Minimal performance gain doesn't justify complexity/speed cost"

In short: the HF model shows a tiny improvement in metrics, but the baseline is much faster in inference (large speed ratio) and is preferred for production by default. To use the HF model in production, set `USE_HF = True` in `app.py`.

Notes about hardware / GPU
-------------------------

- Baseline (scikit-learn) is CPU-friendly and very fast for inference. It is the recommended default for low-latency/cheap deployments.
- Hugging Face models (in `models/hf_model`) benefit significantly from a CUDA GPU for both training and inference. For training, a GPU with at least 8GB of VRAM is recommended for small-medium models (e.g., DistilBERT). If you have multiple GPUs or large datasets, consider using `accelerate` or distributed training.
- For inference on CPU, consider using smaller transformer variants (DistilBERT) or use ONNX/quantization to reduce latency and memory.

How to evaluate new data (example script / notebook cell)
--------------------------------------------------------

Below is a small notebook-style Python snippet you can paste into `notebook.ipynb` or run as a script to evaluate a CSV of labeled examples (expects columns `text` and `label` or auto-detects common names):

```python
# Notebook cell: Evaluate a local model on a CSV file
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def evaluate_baseline(test_csv, text_col='text', label_col='label'):
	df = pd.read_csv(test_csv)
	pipe = joblib.load('models/baseline_model.pkl')
	preds = pipe.predict(df[text_col].astype(str).tolist())
	print('accuracy', accuracy_score(df[label_col], preds))
	print('f1_macro', f1_score(df[label_col], preds, average='macro'))

def evaluate_hf(test_csv, text_col='text', label_col='label', hf_path='models/hf_model'):
	df = pd.read_csv(test_csv)
	tokenizer = AutoTokenizer.from_pretrained(hf_path)
	model = AutoModelForSequenceClassification.from_pretrained(hf_path)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	texts = df[text_col].astype(str).tolist()
	inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=128).to(device)
	with torch.no_grad():
		logits = model(**{k: v for k, v in inputs.items() if k in ['input_ids','attention_mask']}).logits
	preds = logits.argmax(dim=-1).cpu().numpy()
	print('accuracy', accuracy_score(df[label_col], preds))
	print('f1_macro', f1_score(df[label_col], preds, average='macro'))

# Example usage
# evaluate_baseline('data/test.csv')
# evaluate_hf('data/test.csv')
```

If you prefer a small script, adapt the above into `evaluate.py` and run `python evaluate.py data/test.csv`.

Files of interest
-----------------

- `app.py` — FastAPI app and inference endpoint (`/predict`).
- `train.py` — HF training script using transformers Trainer.
- `run_train.ps1` — convenience PowerShell wrapper (if present).
- `notebook.ipynb` — notebook used during experiments.
- `models/` — trained artifacts and `model_selection.json` describing the chosen model.

Try it
------

1. Install deps and start the API (see steps above).
2. Use the curl example to POST text to `/predict`.

Completion summary
------------------

This README provides quick instructions for running locally, example requests, how training was done, why the baseline model was selected for production, hardware notes, and a small evaluation snippet you can run from a notebook or script.

