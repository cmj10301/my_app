from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uuid
import json
import os

# 모델 로드
model_path = "ai/models/q_network_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("q_network_model.h5 파일이 존재하지 않습니다. 먼저 학습을 완료하세요.")
q_network_model = tf.keras.models.load_model(model_path)

# 질문 텍스트 로딩
with open("ai/dataset/final_dataset_with_family_and_size.json", encoding="utf-8") as f:
    questions = json.load(f)["animals"][0]["questions"]
question_texts = [q["question"] for q in questions]
num_questions = len(question_texts)
num_animals = len(json.load(open("ai/dataset/final_dataset_with_family_and_size.json", encoding="utf-8"))["animals"])
total_actions = num_questions + num_animals

# 세션 저장소
session_store = {}

app = FastAPI()

class AnswerRequest(BaseModel):
    session_id: str
    answer: int  # 1: 예, 0: 아니오, -1: 모르겠음

@app.post("/start")
def start_game():
    session_id = str(uuid.uuid4().int)[:8]
    state = [-1.0] * num_questions
    session_store[session_id] = state
    return {
        "session_id": session_id,
        "message": "게임 시작",
        "question": question_texts[0],
        "question_index": 0
    }

@app.post("/answer")
def answer_question(req: AnswerRequest):
    if req.session_id not in session_store:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = session_store[req.session_id]

    # 이전 질문 인덱스 찾기
    last_q_idx = next((i for i, v in enumerate(state) if v == -1), None)
    if last_q_idx is None:
        raise HTTPException(status_code=400, detail="질문이 모두 완료되었습니다.")

    # 상태 업데이트
    state[last_q_idx] = req.answer
    session_store[req.session_id] = state

    # 추론
    input_array = np.array([state], dtype=np.float32)
    q_values = q_network_model.predict(input_array, verbose=0)[0]
    next_action = int(np.argmax(q_values))

    if next_action < num_questions:
        return {
            "session_id": req.session_id,
            "question": question_texts[next_action],
            "question_index": next_action
        }
    else:
        guess_index = next_action - num_questions
        return {
            "session_id": req.session_id,
            "guess": True,
            "guess_index": guess_index,
            "guess_name": f"{guess_index}번 동물 (이름 매핑은 프론트에서 처리 가능)"
        }
