from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import tensorflow as tf, numpy as np, json, os, uuid

# ---------- 데이터 ----------
DATA_PATH = "ai/dataset/final_dataset_with_family_and_size.json"
with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)

animals       = raw["animals"]
animal_names  = [a["name"] for a in animals]
question_text = [q["question"] for q in animals[0]["questions"]]

NQ  = len(question_text)
NA  = len(animal_names)
ACT = NQ + NA

# ---------- 모델 ----------
MODEL_DIR = "ai/models/q_net_saved"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError("SavedModel(q_net_saved/)이 없습니다. 학습을 먼저 진행하세요.")
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures["serving_default"]

# ---------- 상수 ----------
UNASKED = -2.0   # 아직 물어보지 않음
UNKNOWN = -1.0   # “모르겠다”

# ---------- 세션 ----------
class GameState(BaseModel):
    state:   List[float]   # 길이 NQ
    current: int           # 지금 화면에 보이는 질문 번호

SessionStore: Dict[str, GameState] = {}

# ---------- FastAPI ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnswerRequest(BaseModel):
    session_id: str
    answer: int            # 1 / 0 / -1

# ---------- 라우터 ----------
@app.post("/start")
def start_game():
    sid = str(uuid.uuid4().int)[:8]
    SessionStore[sid] = GameState(state=[UNASKED]*NQ, current=0)
    return {
        "session_id": sid,
        "message": "게임 시작",
        "question": question_text[0],
        "question_index": 0
    }

@app.post("/answer")
def answer_question(req: AnswerRequest):
    gs = SessionStore.get(req.session_id)
    if gs is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다.")

    if req.answer not in (1, 0, -1):
        raise HTTPException(400, "answer 값은 1, 0, -1 중 하나여야 합니다.")

    # 1) 답 기록
    gs.state[gs.current] = float(req.answer)

    # 2) 추론
    q_vals = infer(tf.convert_to_tensor([gs.state], tf.float32))["q_values"][0].numpy()

    # 이미 답변한 질문은 아주 작은 값으로
    for i, v in enumerate(gs.state):
        if v != UNASKED:
            q_vals[i] = -1e9

    next_act = int(np.argmax(q_vals))

    # 3) 결과 전송
    if next_act < NQ:                       # 다음 질문
        gs.current = next_act
        SessionStore[req.session_id] = gs   # 갱신
        return {
            "session_id": req.session_id,
            "question": question_text[next_act],
            "question_index": next_act
        }
    else:                                   # 추리
        guess_idx = next_act - NQ
        # 더 이상 진행 필요 없으므로 세션 제거(선택)
        del SessionStore[req.session_id]
        return {
            "session_id": req.session_id,
            "guess": True,
            "guess_index": guess_idx,
            "guess_name": animal_names[guess_idx]
        }
