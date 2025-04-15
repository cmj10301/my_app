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

# 지도학습 모델 불러오기 (필요한 경우 함께 사용)
from tensorflow.keras import models
inference_model = models.load_model(os.path.join("ai", "models", "inference_model.h5"), compile=False)

# 데이터셋 로드
with open(os.path.join("ai", "dataset", "final_dataset_with_family_and_size.json"), 'r', encoding='utf-8') as f:
    data = json.load(f)
    animals = data["animals"]
    unique_animals = [a["name"] for a in animals]

print("Unique animals:", unique_animals)


# -------------------------------
# 환경 정의 (TwentyQuestionsTFEnv)
# -------------------------------
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
        # target 동물을 랜덤으로 선택합니다.
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


# -------------------------------
# 환경 및 에이전트 구성 (강화학습 부분)
# -------------------------------
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

# 초기 경험 수집 (자동으로 100번 진행)
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


# -------------------------------
# 자동화 에피소드 생성 (simulate_episode)
# -------------------------------
def simulate_episode(env, target_animal):
    """
    주어진 target_animal(데이터셋의 동물 객체) 정보를 이용하여
    자동으로 정답(사용자 정답처럼)을 입력하는 에피소드를 생성합니다.
    """
    # env.reset()를 호출하면 랜덤 target이 선택되지만, 여기서는 우리가 지정한 target_animal으로 덮어씁니다.
    env.target = target_animal
    env.asked_questions = set()
    # 정답 벡터(각 질문에 대한 정답)을 가져옵니다.
    correct_answers = [q['answer'] for q in target_animal['questions']]
    env.history = correct_answers.copy()
    # 모든 질문을 이미 “답변”한 것으로 처리합니다.
    env.asked_questions = set(range(env.num_questions))
    env._episode_ended = True
    # 최종 추측 (정답으로 추측)
    final_time_step = env.guess(target_animal['name'])
    # 생성된 경험(trajectory)을 반환합니다.
    batched_state = np.array([env.history], dtype=np.float32)
    user_time_step = ts.TimeStep(
        step_type=np.array([ts.StepType.FIRST], dtype=np.int32),
        reward=np.array([0.0], dtype=np.float32),
        discount=np.array([1.0], dtype=np.float32),
        observation=batched_state)
    user_next_time_step = ts.TimeStep(
        step_type=np.array([ts.StepType.LAST], dtype=np.int32),
        reward=np.array([float(final_time_step.reward)], dtype=np.float32),
        discount=np.array([0.0], dtype=np.float32),
        observation=batched_state)
    dummy_action = np.array([0], dtype=np.int32)
    dummy_action_step = policy_step.PolicyStep(action=dummy_action, state=())
    user_traj = trajectory.from_transition(user_time_step, dummy_action_step, user_next_time_step)
    return user_traj, final_time_step

# -------------------------------
# 자동 학습 루프: 시뮬레이션 에피소드로 학습하기
# -------------------------------
def automated_training(num_episodes=1000, steps_per_episode=100):
    """
    데이터셋의 동물들을 순회하거나 무작위로 선택하여
    자동으로 에피소드를 생성하고, 이를 Replay Buffer에 추가한 후,
    추가 경험을 이용해 강화학습 에이전트를 학습합니다.
    """
    num_animals = len(animals)
    for ep in range(num_episodes):
        # 동물들을 무작위로 선택하거나 순서대로 선택할 수 있습니다.
        target_animal = random.choice(animals)
        traj, final_ts = simulate_episode(train_py_env, target_animal)
        # Replay Buffer에 추가
        replay_buffer.add_batch(traj)
        # (선택 사항) 에피소드 당 일정 횟수의 학습 진행
        for _ in range(steps_per_episode):
            experience, _ = next(iterator)
            loss_info = agent.train(experience)
        if (ep+1) % 100 == 0:
            ckpt_path = checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))
            print(f"에피소드 {ep+1} 진행, checkpoint saved at {ckpt_path}. Loss: {loss_info.loss.numpy()}")
    print("자동 학습 완료!")

# -------------------------------
# 사용자와 상호작용하는 통합 추론 함수 (선택 사항)
# -------------------------------
def integrated_inference():
    """
    사용자가 생각한 동물을 맞추기 위해, AI는 질문을 통해 후보 목록을 좁힙니다.
    (여기서는 사용자 입력을 받는 부분도 포함되어 있으나, 자동 학습과 별개로 추론 후 추가 경험을 저장합니다.)
    """
    print("사용자가 생각하는 동물을 맞추겠습니다.")
    correct_animal = input("사용자가 생각한 동물(정답)을 입력하세요: ").strip()
    print("----- 질문을 시작합니다. -----")
    state = train_env.reset()  # 배치 관측값 (1, num_questions)
    asked = set()
    guess_animal = None  # 후보 단계에서 결정된 추측값

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
        
        train_py_env.history[q_index] = user_answer
        train_py_env.asked_questions.add(q_index)
        state = ts.transition(np.array([train_py_env.history], dtype=np.float32),
                              reward=-1.0,
                              discount=1.0)

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

            print("----------------------------")
            print("현재 후보 동물:", candidates)
            print("현재 답변 벡터:", train_py_env.history[:train_py_env.num_questions])
            print("----------------------------")
            if len(candidates) == 1 and confidence >= 0.9:
                print("후보 목록이 1개로 좁혀졌습니다.")
                guess_animal = candidates[0]
                break
            elif confidence >= 0.9 and len(candidates) <= 3:
                for idx, candidate in enumerate(candidates):
                    print(f"{idx+1}. {candidate}")
                choice = input("이 중 올바른 동물을 선택하세요 (번호 입력, 엔터를 누르면 계속 질문합니다): ").strip()
                if choice != "":
                    try:
                        choice = int(choice)
                        if 1 <= choice <= len(candidates):
                            guess_animal = candidates[choice - 1]
                            break
                        else:
                            print("잘못된 선택입니다. 계속 질문 진행.")
                    except Exception:
                        print("입력 오류입니다. 계속 질문 진행.")
    print("----- 질문 종료 -----")
    print("입력한 답변 벡터:", train_py_env.history[:train_py_env.num_questions])
    
    final_guess = input("최종 추측할 동물명을 입력하세요 (엔터: 후보 또는 지도학습 모델 사용): ").strip()
    if final_guess == "":
        if guess_animal is not None:
            final_guess = guess_animal
        else:
            inference_input = np.array(train_py_env.history[:train_py_env.num_questions], dtype=np.float32).reshape(1, -1)
            pred_probs = inference_model.predict(inference_input)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            final_guess = unique_animals[pred_class]
            print("지도학습 모델 최종 예측 동물:", final_guess)
    
    final_time_step = train_py_env.guess(final_guess)
    print("추측 결과 보상:", float(final_time_step.reward))
    print("실제 정답 동물:", correct_animal)
    
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
    
    print("추가 경험을 이용하여 100 스텝의 학습을 진행합니다...")
    for i in range(100):
        experience, _ = next(iterator)
        loss_info = agent.train(experience)
        if (i + 1) % 100 == 0:
            ckpt_path = checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))
            print(f"Step {agent.train_step_counter.numpy()} checkpoint saved at {ckpt_path}. Loss: {loss_info.loss.numpy()}")
    print("추가 학습 완료!")


# -------------------------------
# 실행 선택
# -------------------------------
if __name__ == "__main__":
    mode = input("자동 학습을 진행하려면 'auto'를, 사용자 입력을 받으려면 그냥 엔터를 누르세요: ").strip().lower()
    if mode == "auto":
        print("자동 학습을 시작합니다.")
        automated_training(num_episodes=1000, steps_per_episode=100)
    else:
        integrated_inference()
