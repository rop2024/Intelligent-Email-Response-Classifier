# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class Input(BaseModel):
    text: str

# Choose which model to load
USE_HF = True  # set False to use baseline pipeline

app = FastAPI(title="Reply Classifier")

if USE_HF:
    hf_path = "models/hf_model"
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    model = AutoModelForSequenceClassification.from_pretrained(hf_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    pipe = joblib.load("models/baseline_model.pkl")

with open("models/id2label.json") as f:
    id2label = json.load(f)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum(axis=-1, keepdims=True)

@app.post("/predict")
def predict(item: Input):
    text = item.text
    if USE_HF:
        enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.cpu().numpy()[0]
        probs = softmax(logits)
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
    else:
        pred_id = int(pipe.predict([text])[0])
        probs = pipe.predict_proba([text])[0]
        confidence = float(max(probs))

    return {"label": id2label[str(pred_id)], "confidence": round(confidence,3)}

