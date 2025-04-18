from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ เปิด CORS สำหรับ frontend ที่ localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://emotion-classification-ashen.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# โหลด tokenizer + model
model_name = "bypkt/multi-emotion-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
emotion_names = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    text = body.get("text")
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
