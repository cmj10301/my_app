#uvicorn pages.api.main:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle


app = FastAPI()

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"]만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("ai/models/model.h5")
with open("ai/models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


class QuestionAnswers(BaseModel):
    answers: list[int]

@app.post("/predict")
def predict_character(data: QuestionAnswers):
    answers = data.answers
    padded = answers + [3] * (20 - len(answers))  # 패딩 값은 모델 학습 시 설정한 것과 같게
    input_array = np.array([padded])
    
    pred = model.predict(input_array)[0]
    top_prob = np.max(pred)
    idx = np.argmax(pred)
    name = label_encoder.inverse_transform([idx])[0]

    # 확신도 기준: 0.9 이상이면 예측 확정
    if top_prob > 0.6 or len(answers) >= 20:
        return {"result": name, "confidence": float(top_prob)}
    else:
        return {"result": None}