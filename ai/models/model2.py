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
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
import json

# ------------------ 환경 정의 ------------------
class TwentyQuestionsTFEnv(py_environment.PyEnvironment):
    def __init__(self, dataset):
        self.dataset = dataset  # 동물 데이터셋: 각 항목은 {"name": ..., "questions": [...]}
        self.num_questions = len(dataset[0]['questions'])
        # 질문 액션: 0 ~ num_questions-1 (질문 인덱스)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_questions - 1, name='action')
        # 관측(observation): 질문 num_questions개에 대한 응답 상태 (-1: 미답변, 그 외 1, 0, 2)
        # unbatched 관측값의 shape는 (num_questions,) 즉, (20,)
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
        self.history = [-1] * self.num_questions  # 초기에는 답변 없음 (-1)
        self._episode_ended = False
        # 반환할 관측값은 unbatched, shape: (num_questions,)
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

    # ---------- 추측 액션 추가 ----------
    def guess(self, guess_name):
        if not self._episode_ended:
            self._episode_ended = True
        if guess_name == self.target['name']:
            reward = 100.0
        else:
            reward = -50.0
        return ts.termination(np.array(self.history, dtype=np.float32), reward=reward)


# ------------------ 데이터셋 불러오기 ------------------
with open('ai/dataset/final_dataset_with_family_and_size.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    animals = data["animals"]

# ------------------ TF-Agents 환경 및 에이전트 구성 ------------------
# TFPyEnvironment는 내부 환경의 unbatched 관측값을 받아 배치 차원을 추가합니다.
train_py_env = TwentyQuestionsTFEnv(animals)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),  # 예상 shape: (20,)
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),  # 이때 observation는 (1,20)
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

# 데이터 수집 (초기 경험)
collect_steps_per_iteration = 100
for _ in range(collect_steps_per_iteration):
    collect_step(train_env, random_policy, replay_buffer)

dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)

# ------------------ 체크포인트 설정 및 로드 ------------------
checkpoint_dir = "ai/checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer, train_step_counter=train_step_counter)
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    checkpoint.restore(latest_ckpt)
    print(f"Checkpoint {latest_ckpt} 복원됨.")
    
# 자동 학습 루프 (기본 학습)
num_iterations = 2000  # 학습 반복 횟수
for i in range(num_iterations):
    experience, _ = next(iterator)
    loss_info = agent.train(experience)
    if agent.train_step_counter.numpy() % 1000 == 0:
        ckpt_path = checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))
        print(f"Step {agent.train_step_counter.numpy()} checkpoint saved at {ckpt_path}.")
print("기본 학습 완료!")

# ------------------ 인터랙티브 게임 및 추가 학습 ------------------
# --- 인터랙티브 게임 및 추가 학습 ---
def interactive_game():
    """
    사용자가 질문에 답변하면서, 일정 조건(예: 5개 이상의 질문)이 충족되면
    각 동물과의 답변 일치도를 계산하여 신뢰도가 90% 이상일 경우 자동 추측 후보를 제시합니다.
    후보가 3개 이하일 때만 후보 목록을 표시하여 번호로 선택하게 하고,
    후보가 3개 초과면 아무 후보도 선택하지 않고 질문을 이어갑니다.
    """
    print("----- 20개 질문에 답해주세요 -----")
    # TFPyEnvironment를 사용하므로, train_env.reset()은 내부에서 배치 차원을 추가하여 (1, num_questions)를 반환합니다.
    state = train_env.reset()  
    asked = set()

    guess_animal = None  # 초기화

    while True:
        available = set(range(train_py_env.num_questions)) - asked
        if len(available) == 0:
            print("모든 질문을 마쳤습니다.")
            break

        # 에이전트 정책으로 질문 인덱스 결정 (state는 (1, num_questions))
        action_step = agent.policy.action(state)
        q_index = int(np.squeeze(action_step.action))
        if q_index not in available:
            q_index = random.choice(list(available))
        asked.add(q_index)
        
        # 질문 텍스트 출력 (여기서는 첫 번째 동물의 질문 목록을 사용)
        question_text = train_py_env.dataset[0]['questions'][q_index]['question']
        print(f"질문 {q_index+1}: {question_text}")
        
        user_input = input("당신의 답변 (1=예, 0=아니오, 2=모르겠다): ").strip()
        try:
            user_answer = int(user_input)
            if user_answer not in [0, 1, 2]:
                print("잘못된 입력입니다. -1 (미답변)으로 처리합니다.")
                user_answer = -1
        except Exception:
            print("입력 오류 발생, -1 (미답변)으로 처리합니다.")
            user_answer = -1
        
        # 환경 상태 업데이트 (history 업데이트; 이 history는 unbatched, shape=(num_questions,))
        train_py_env.history[q_index] = user_answer
        train_py_env.asked_questions.add(q_index)
        
        # 상태 업데이트: unbatched history를 np.array([history])로 만들어 (1, num_questions)
        state = ts.transition(np.array([train_py_env.history], dtype=np.float32),
                              reward=-1.0,
                              discount=1.0)
        
        # 자동 추측 판단: 답변된 질문 수가 5개 이상이면 각 동물과의 일치도를 계산
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
            # 만약 신뢰도가 90% 이상이고 후보 수가 3개 이하이면 후보 목록 표시
            if confidence >= 0.9 and len(candidates) <= 3:
                print("AI가 충분히 확신합니다.")
                print("자동 추측 후보 (동점):")
                for idx, candidate in enumerate(candidates):
                    print(f"{idx+1}. {candidate}")
                choice = input("이 중에서 올바른 동물을 선택하세요 (번호 입력): ").strip()
                try:
                    choice = int(choice)
                    if 1 <= choice <= len(candidates):
                        guess_animal = candidates[choice - 1]
                        break  # 추측 액션 실행
                    else:
                        print("잘못된 선택입니다. 계속 질문을 진행합니다.")
                except Exception:
                    print("입력 오류. 계속 질문을 진행합니다.")
            elif confidence >= 0.9:
                # 후보가 3개 초과이면 후보 목록은 표시하지 않고, 질문 계속 진행
                print("AI의 신뢰도가 충분하지만, 후보 수가 많아 자동 추측 후보를 확정하지 않습니다.")
                # continue 질문 진행
        
    print("----- 질문 종료 -----")
    print("입력한 답변 벡터:", train_py_env.history)
    
    # 만약 자동 추측 후보가 확정되지 않았다면, 사용자가 직접 입력
    if guess_animal is None:
        guess_animal = input("추측할 동물명을 입력하세요: ").strip()
    
    final_time_step = train_py_env.guess(guess_animal)
    print("추측 결과 보상:", float(final_time_step.reward))
    print("실제 정답 동물:", train_py_env.target['name'])
    
    # --- 사용자 경험을 Replay Buffer에 단 한 번 추가한 후 추가 학습 수행 ---
    # final_state: unbatched, shape (num_questions,) — 새 데이터셋 기준 예: (23,)
    final_state = np.array(train_py_env.history, dtype=np.float32)
    # 여기서 배치 차원이 없어야 하므로 final_state 그대로 사용합니다.
    user_time_step = ts.restart(final_state)  # TimeStep with observation shape: (num_questions,)
    # 대신, 에이전트의 정책 함수 호출 대신 더미 액션을 생성합니다.
    dummy_action = np.array(0, dtype=np.int32)  # action spec: scalar
    dummy_action_step = policy_step.PolicyStep(action=dummy_action, state=())
    user_next_time_step = ts.termination(final_state, reward=float(final_time_step.reward))
    user_traj = trajectory.from_transition(user_time_step, dummy_action_step, user_next_time_step)
    replay_buffer.add_batch(user_traj)
    experience, _ = next(iterator)
    loss_info = agent.train(experience)
    ckpt_path = checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))
    print(f"추가 학습 후 checkpoint 저장됨: {ckpt_path}")


# --- 인터랙티브 게임 실행 ---
interactive_game()
