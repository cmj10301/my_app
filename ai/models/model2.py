import os
import random
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import csv

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tensorflow.keras import layers, models

# 모델 및 데이터 로딩
inference_model = models.load_model("ai/models/inference_model.h5", compile=False)
with open("ai/dataset/final_dataset_with_family_and_size.json", encoding='utf-8') as f:
    data = json.load(f)
    animals = data["animals"]

# 환경 정의
class TwentyQuestionsTFEnv(py_environment.PyEnvironment):
    def __init__(self, dataset, noise_rate=0.05):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_questions = len(dataset[0]['questions'])
        self.unique_animals = [a["name"] for a in dataset]
        self.num_animals = len(self.unique_animals)
        self.total_actions = self.num_questions + self.num_animals
        self._action_spec = array_spec.BoundedArraySpec((), np.int32, 0, self.total_actions - 1)
        self._observation_spec = array_spec.BoundedArraySpec((self.num_questions,), np.float32, -1, 2)
        self._episode_ended = False
        self.reset()

    def action_spec(self): return self._action_spec
    def observation_spec(self): return self._observation_spec

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

        if action < self.num_questions:
            if action in self.asked_questions and train_step_counter.numpy() < 10000:
                return ts.transition(np.array(self.history, dtype=np.float32), reward=-1.0, discount=1.0)

            self.asked_questions.add(action)
            answer = self.target['questions'][action]['answer']

            # ✨ 수정된 부분: 질문 1개당 -1.0 페널티 부여
            question_penalty = -1.0
            self.history[action] = answer
            return ts.transition(np.array(self.history, dtype=np.float32), reward=question_penalty, discount=1.0)

        else:
            if len(self.asked_questions) < 5:
                return ts.transition(np.array(self.history, dtype=np.float32), reward=-10.0, discount=1.0)

            guess_index = action - self.num_questions
            guess_name = self.unique_animals[guess_index]
            self._episode_ended = True
            reward = 100.0 if guess_name == self.target['name'] else -50.0
            return ts.termination(np.array(self.history, dtype=np.float32), reward=reward)


# 에이전트 구성
train_py_env = TwentyQuestionsTFEnv(animals)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

fc_layer_params = (64,)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    preprocessing_layers=None,
    preprocessing_combiner=None,
    fc_layer_params=fc_layer_params
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
epsilon = tf.Variable(1.0)

# epsilon 0~5000 스텝 동안 감소
def decay_epsilon():
    step = train_step_counter.numpy()
    epsilon_value = np.interp(step, [0, 5000], [1.0, 0.1])
    epsilon.assign(epsilon_value)

def get_epsilon():
    decay_epsilon()
    return epsilon.numpy()

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_net,
    optimizer,
    td_errors_loss_fn=common.element_wise_huber_loss,
    train_step_counter=train_step_counter,
    epsilon_greedy=get_epsilon
)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    train_env.batch_size,
    max_length=10000
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_info = tf.config.experimental.get_device_details(gpus[0])
    gpu_name = gpu_info.get('device_name', 'Unknown GPU')
    if "4060" in gpu_name:
        sample_batch_size = 256
    else:
        sample_batch_size = 128
else:
    sample_batch_size = 64

dataset = replay_buffer.as_dataset(
    sample_batch_size=sample_batch_size,
    num_steps=2
).prefetch(3)
iterator = iter(dataset)

# 체크포인트 설정
checkpoint_dir = "ai/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer, train_step_counter=train_step_counter)
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    checkpoint.restore(latest_ckpt)

# 기록용 리스트
avg_rewards, question_counts, ckpt_log = [], [], []

# 에피소드 진행
def simulate_episode_with_guess(env, policy, buffer):
    time_step = env.reset()
    total_reward = 0.0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        buffer.add_batch(trajectory.from_transition(time_step, action_step, next_time_step))
        time_step = next_time_step
        total_reward += time_step.reward
    avg_rewards.append(float(total_reward))
    question_counts.append(len(env._envs[0].asked_questions))

# 학습 함수
def automated_training(num_episodes=15000, steps_per_episode=300):
    log_rows = []
    correct = 0
    for ep in range(num_episodes):
        simulate_episode_with_guess(train_env, agent.collect_policy, replay_buffer)

        if replay_buffer.num_frames() > 0:
            for _ in range(steps_per_episode):
                experience, _ = next(iterator)
                agent.train(experience)

        if train_py_env._episode_ended and train_py_env.target['name'] in train_py_env.unique_animals:
            guess_index = np.argmax(train_py_env.history)
            guess_name = train_py_env.unique_animals[guess_index] if guess_index < len(train_py_env.unique_animals) else ""
            if guess_name == train_py_env.target['name']:
                correct += 1

        total_q = question_counts[-1]
        acc = correct / (ep + 1)
        log_rows.append((ep + 1, avg_rewards[-1], total_q, acc))

        if (ep + 1) % 500 == 0:
            print(f"[에피소드 {ep+1}] 최근 평균 보상: {np.mean(avg_rewards[-500:]):.2f}, 최근 평균 질문 수: {np.mean(question_counts[-500:]):.2f}, 정확도: {acc*100:.2f}%")

        if (ep + 1) % 500 == 0:
            ckpt_path = checkpoint.save(os.path.join(checkpoint_dir, "ckpt"))
            ckpt_log.append((ep + 1, avg_rewards[-1], total_q, os.path.basename(ckpt_path)))

    with open("ai/log/reward_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "question_count", "accuracy", "ckpt_file"])
        for i, row in enumerate(ckpt_log):
            ep, r, q, ckpt = row
            acc = log_rows[i][3]
            writer.writerow([ep, r, q, acc, ckpt])

    plt.plot(avg_rewards, label="보상")
    plt.plot(question_counts, label="질문 수")
    plt.plot([x[3] * 100 for x in log_rows], label="정답률 (%)")
    plt.xlabel("에피소드")
    plt.ylabel("값")
    plt.title("학습 추이 (보상 / 질문 수 / 정답률)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    q_net.model.save('ai/models/q_network_model.h5')
    print("✅ 학습 완료! Q-Network 모델 저장되었습니다.")

# 평가 함수
def evaluate_accuracy(env, policy, test_episodes=100):
    correct, total_questions = 0, 0
    for _ in range(test_episodes):
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
        total_questions += len(env._envs[0].asked_questions)
        if time_step.reward.numpy() >= 100.0:
            correct += 1
    acc = correct / test_episodes
    avg_q = total_questions / test_episodes
    print(f"✅ 정확도: {acc * 100:.2f}% (평균 질문 수: {avg_q:.2f})")

if __name__ == "__main__":
    automated_training(num_episodes=15000, steps_per_episode=300)
