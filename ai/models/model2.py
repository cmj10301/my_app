# ai/models/model2.py

import os
import random
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment, tf_py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, policy_step
import json

# 지도학습 모델 불러오기
from tensorflow.keras import models
inference_model = models.load_model(os.path.join("ai", "models", "inference_model.h5"))

# 데이터셋 로드
with open(os.path.join("ai", "dataset", "final_dataset_with_family_and_size.json"), 'r', encoding='utf-8') as f:
    data = json.load(f)
    animals = data["animals"]
    unique_animals = [a["name"] for a in animals]

print("Unique animals:", unique_animals)

# 환경 정의
class TwentyQuestionsTFEnv(py_environment.PyEnvironment):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_questions = len(dataset[0]['questions'])
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_questions - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.num_questions,), dtype=np.float32, minimum=-1, maximum=2, name='observation')
        self._episode_ended = False
        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.target = random.choice(self.dataset)
        self.asked_questions = set()
        self.history = [-1] * self.num_questions
        self._episode_ended = False
        return ts.restart(np.array(self.history, dtype=np.float32))

    def _step(self, action):
        action = int(np.squeeze(action))
        if self._episode_ended:
            return self.reset()
        if action in self.asked_questions:
            reward = -5.0
            return ts.transition(np.array(self.history, dtype=np.float32), reward=reward, discount=1.0)
        self.asked_questions.add(action)
        answer = self.target['questions'][action]['answer']
        self.history[action] = answer
        reward = -1.0
        if len(self.asked_questions) == self.num_questions:
            self._episode_ended = True
        return ts.transition(np.array(self.history, dtype=np.float32), reward=reward, discount=1.0)

    def guess(self, guess_name):
        if not self._episode_ended:
            self._episode_ended = True
        reward = 100.0 if guess_name == self.target['name'] else -50.0
        return ts.termination(np.array(self.history, dtype=np.float32), reward=reward)

# 환경 및 에이전트 구성 (강화학습 부분)
train_py_env = TwentyQuestionsTFEnv(animals)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

fc_layer_params = (100,)
q_net = q_network.QNetwork(train_env.observation_spec(),
                           train_env.action_spec(),
                           fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                           train_env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_huber_loss,
                           train_step_counter=train_step_counter)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=10000)

from tf_agents.policies import random_tf_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

# 초기 경험 수집
for _ in range(100):
    collect_step(train_env, random_policy, replay_buffer)

dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)

checkpoint_dir = "ai/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer, train_step_counter=train_step_counter)
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    checkpoint.restore(latest_ckpt)
    print(f"Checkpoint {latest_ckpt} 복원됨.")

# 자동 학습 루프 (기본 학습)
# num_iterations = 2000  # 학습 반복 횟수
# for i in range(num_iterations):
#     experience, _ = next(iterator)
#     loss_info = agent.train(experience)
#     if agent.train_step_counter.numpy() % 1000 == 0:
#         ckpt_path = checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))
#         print(f"Step {agent.train_step_counter.numpy()} checkpoint saved at {ckpt_path}.")
# print("기본 학습 완료!")

# 통합 추론 함수
def integrated_inference():
    """
    사용자가 질문에 답변한 후, 두 경로(강화학습 기반 및 지도학습 분류 모델)를 통해 결과를 출력합니다.
    - 지도학습 모델: 사용자의 답변 벡터를 입력받아 동물 클래스를 예측합니다.
    - RL 기반: 사용자의 추측 및 보상을 산출합니다.
    질문은 후보 리스트가 1개만 남거나, 후보가 여러 개일 경우 사용자가 선택할 수 있게 합니다.
    """
    print("----- 20개 질문에 답해주세요 -----")
    state = train_env.reset()  # 배치 관측값 (1, num_questions)
    asked = set()
    guess_animal = None  # 최종 추측 동물을 저장할 변수

    while True:
        available = set(range(train_py_env.num_questions)) - asked
        if len(available) == 0:
            print("모든 질문을 마쳤습니다.")
            break

        action_step = agent.policy.action(state)
        q_index = int(np.squeeze(action_step.action))
        if q_index not in available:
            q_index = random.choice(list(available))
        asked.add(q_index)
        
        question_text = train_py_env.dataset[0]['questions'][q_index]['question']
        print(f"질문 {q_index+1}: {question_text}")
        try:
            user_input = input("당신의 답변 (1=예, 0=아니오, 2=모르겠다): ").strip()
            user_answer = int(user_input)
            if user_answer not in [0, 1, 2]:
                print("잘못된 입력입니다. -1 (미답변)으로 처리합니다.")
                user_answer = -1
        except Exception:
            print("입력 오류 발생, -1 (미답변)으로 처리합니다.")
            user_answer = -1
        
        # 환경 상태 업데이트 (history 업데이트)
        train_py_env.history[q_index] = user_answer
        train_py_env.asked_questions.add(q_index)
        state = ts.transition(np.array([train_py_env.history], dtype=np.float32),
                              reward=-1.0,
                              discount=1.0)

        # 최소 답변 수가 5개 이상이면 후보 리스트를 계산합니다.
        answered_count = sum(1 for x in train_py_env.history if x != -1)
        if answered_count >= 5:
            scores = {}
            for animal in train_py_env.dataset:
                score = 0
                for i, ans in enumerate(train_py_env.history):
                    if ans == -1:
                        continue
                    if ans == animal['questions'][i]['answer']:
                        score += 1
                scores[animal['name']] = score
            max_score = max(scores.values())
            candidates = [name for name, score in scores.items() if score == max_score]
            confidence = max_score / answered_count if answered_count > 0 else 0

            # 후보 리스트가 한 개만 남았고, 신뢰도가 충분하면 자동 추측
            if len(candidates) == 1 and confidence >= 0.9:
                print("AI가 충분히 확신합니다.")
                print("자동 추측 후보:", candidates[0])
                guess_animal = candidates[0]
                break
            # 후보가 3개 이하인데 신뢰도가 0.9 이상이면 후보 목록 제시
            elif confidence >= 0.9 and len(candidates) <= 3:
                print("AI가 충분히 확신합니다.")
                print("자동 추측 후보 (동점):")
                for idx, candidate in enumerate(candidates):
                    print(f"{idx+1}. {candidate}")
                choice = input("이 중에서 올바른 동물을 선택하세요 (번호 입력, 엔터를 누르면 계속 질문합니다): ").strip()
                if choice != "":
                    try:
                        choice = int(choice)
                        if 1 <= choice <= len(candidates):
                            guess_animal = candidates[choice - 1]
                            break
                        else:
                            print("잘못된 선택입니다. 계속 질문을 진행합니다.")
                    except Exception:
                        print("입력 오류. 계속 질문을 진행합니다.")
            # 조건이 만족되지 않으면 질문 계속 진행.
    
    print("----- 질문 종료 -----")
    print("입력한 답변 벡터:", train_py_env.history[:train_py_env.num_questions])
    
    # 지도학습 모델 추론
    inference_input = np.array(train_py_env.history[:train_py_env.num_questions], dtype=np.float32).reshape(1, -1)
    pred_probs = inference_model.predict(inference_input)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    pred_animal = unique_animals[pred_class]
    print("지도학습 모델 예측 동물:", pred_animal)
    print("각 클래스 확률:", pred_probs)
    
    # 최종 추측: 사용자가 입력하지 않으면 지도학습 모델 예측값을 사용
    final_guess = input("추측할 동물명을 입력하세요 (엔터: 지도학습 모델 예측 사용): ").strip()
    if final_guess == "":
        final_guess = pred_animal
    # 만약 후보 선택 단계에서 이미 자동 추측이 이루어졌다면 우선 그 값을 사용
    if guess_animal is not None:
        final_guess = guess_animal

    final_time_step = train_py_env.guess(final_guess)
    print("추측 결과 보상:", float(final_time_step.reward))
    print("실제 정답 동물:", train_py_env.target['name'])
    
    # 사용자 경험(trajectory) Replay Buffer에 추가 (모든 텐서를 배치화)
    batched_final_state = np.array([train_py_env.history[:train_py_env.num_questions]], dtype=np.float32)
    user_time_step = ts.TimeStep(
        step_type=np.array([ts.StepType.FIRST], dtype=np.int32),
        reward=np.array([0.0], dtype=np.float32),
        discount=np.array([1.0], dtype=np.float32),
        observation=batched_final_state)
    user_next_time_step = ts.TimeStep(
        step_type=np.array([ts.StepType.LAST], dtype=np.int32),
        reward=np.array([float(final_time_step.reward)], dtype=np.float32),
        discount=np.array([0.0], dtype=np.float32),
        observation=batched_final_state)
    dummy_action = np.array([0], dtype=np.int32)
    dummy_action_step = policy_step.PolicyStep(action=dummy_action, state=())
    user_traj = trajectory.from_transition(user_time_step, dummy_action_step, user_next_time_step)
    replay_buffer.add_batch(user_traj)
    print("사용자 경험이 Replay Buffer에 추가되었습니다.")

# 통합 추론 실행
integrated_inference()
