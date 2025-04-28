# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import json
import random

app = FastAPI()

# 모델 및 데이터 불러오기
inference_model = models.load_model("ai/models/inference_model.h5", compile=False)
with open("ai/dataset/final_dataset_with_family_and_size.json", encoding='utf-8') as f:
    data = json.load(f)
    animals = data["animals"]

# 세션별 게임 상태 관리
class GameState:
    def __init__(self):
        self.history = [-1] * len(animals[0]['questions'])
        self.asked_questions = set()
        self.done = False

sessions = {}  # session_id -> GameState 매핑

class UserResponse(BaseModel):
    session_id: str
    user_answer: int  # 0 (아니오), 1 (예)

@app.post("/start")
def start_game():
    session_id = str(random.randint(100000, 999999))  # 간단한 세션 ID
    sessions[session_id] = GameState()
    return {"session_id": session_id, "message": "게임 시작!", "next_question": 0}  # 0번 질문부터 시작

@app.post("/answer")
def answer_question(response: UserResponse):
    if response.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[response.session_id]

    if state.done:
        return {"message": "게임이 이미 끝났습니다. 새로운 게임을 시작하세요."}

    # 현재 질문 인덱스
    next_question = len(state.asked_questions)
    if next_question >= len(state.history):
        state.done = True
        return {"message": "질문이 모두 끝났습니다. 정답을 맞춰보세요."}

    # 유저 답변 반영
    state.history[next_question] = response.user_answer
    state.asked_questions.add(next_question)

    # 추측 시도
    if len(state.asked_questions) >= 5:  # 최소 5개 질문 이후에만 추측
        prediction = inference_model.predict(np.array([state.history]), verbose=0)
        top_index = np.argmax(prediction[0])
        predicted_animal = sorted([a["name"] for a in animals])[top_index]
        return {
            "message": "추측합니다!",
            "predicted_animal": predicted_animal,
            "confidence": float(prediction[0][top_index])
        }
    else:
        # 다음 질문 요청
        return {
            "message": "다음 질문으로 넘어갑니다.",
            "next_question": next_question + 1  # 다음 질문 번호
        }
