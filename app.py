from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://emotion-classification-ashen.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_names = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

@app.on_event("startup")
def load_model():
    global tokenizer, model
    model_name = "bypkt/multi-emotion-RoBERTa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

@app.post("/predict")
async def predict(request: Request):
    if model is None or tokenizer is None:
        return {"error": "Model not ready yet. Try again shortly."}

    body = await request.json()
    text = body.get("text", "")
    if not text.strip():
        return {"error": "Text is empty."}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs_tensor = torch.sigmoid(outputs.logits)

    probs = probs_tensor.squeeze().tolist()
    if isinstance(probs, float):
        probs = [probs]

    labels = [emotion_names[i] for i, prob in enumerate(probs) if prob > 0.5]
    return {
        "text": text,
        "emotion_scores": probs,
        "emotion_label": labels
    }
