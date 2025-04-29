# c:/Users/user_me/Documents/my_app/ai/models/model2.py
# tensorboard 실행: tensorboard --logdir=ai/logs/tensorboard --port=6006

import os, random, json, csv
import numpy as np
import tensorflow as tf

# ── Matplotlib 설정 : 한글 + GUI OFF ──────────────────
import matplotlib
matplotlib.use("Agg")                          # 창을 띄우지 않고 파일만 저장
matplotlib.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts, trajectory
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tensorflow.summary import create_file_writer

# ── 데이터 로드 ───────────────────────────────────────
with open("ai/dataset/final_dataset_with_family_and_size.json", encoding="utf-8") as f:
    animals = json.load(f)["animals"]

# ── 환경 정의 ─────────────────────────────────────────
class TwentyQuestionsTFEnv(py_environment.PyEnvironment):
    def __init__(self, dataset, noise_rate=0.0):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_questions = len(dataset[0]["questions"])
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
        if self._episode_ended: return self.reset()

        # ─ 질문 행동 ─
        if action < self.num_questions:
            if action in self.asked_questions:
                return ts.transition(np.array(self.history, dtype=np.float32), reward=-1.0, discount=1.0)
            self.asked_questions.add(action)
            ans = self.target["questions"][action]["answer"]
            if random.random() < self.noise_rate: ans = 1 - ans
            self.history[action] = ans
            return ts.transition(np.array(self.history, dtype=np.float32), reward=-0.5, discount=1.0)

        # ─ 추측 행동 ─
        if len(self.asked_questions) < 5:
            return ts.transition(np.array(self.history, dtype=np.float32), reward=-20.0, discount=1.0)

        guess_name = self.unique_animals[action - self.num_questions]
        self._episode_ended = True
        reward = 150.0 if guess_name == self.target["name"] else -75.0
        return ts.termination(np.array(self.history, dtype=np.float32), reward=reward)

# ── 에이전트 & 네트워크 ───────────────────────────────
env_py  = TwentyQuestionsTFEnv(animals)
env     = tf_py_environment.TFPyEnvironment(env_py)

q_net = q_network.QNetwork(env.observation_spec(), env.action_spec(), fc_layer_params=(64,))
optimizer = tf.keras.optimizers.Adam(1e-3)
train_step_counter = tf.Variable(0)

def eps_fn():
    step = int(train_step_counter)
    return np.interp(step, [0, 2000], [1.0, 0.1])   # 빠른 ε 감소

agent = dqn_agent.DqnAgent(
    env.time_step_spec(), env.action_spec(),
    q_net, optimizer,
    td_errors_loss_fn = common.element_wise_huber_loss,
    train_step_counter = train_step_counter,
    epsilon_greedy = eps_fn
)
agent.initialize()

# ── ReplayBuffer ─────────────────────────────────────
buffer   = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec, env.batch_size, max_length=5000)
dataset  = buffer.as_dataset(sample_batch_size=256, num_steps=2).prefetch(2)
iterator = iter(dataset)

# ── 로그/경로 ─────────────────────────────────────────
os.makedirs("ai/checkpoints",      exist_ok=True)
os.makedirs("ai/logs/tensorboard", exist_ok=True)
summary_writer = create_file_writer("ai/logs/tensorboard")

# ── 학습 함수 ─────────────────────────────────────────
def automated_training(num_episodes:int, steps_per_ep:int):
    rewards, questions = [], []

    for ep in range(1, num_episodes+1):
        # ① 데이터 수집
        t = env.reset()
        while not t.is_last():
            a = agent.collect_policy.action(t)
            n = env.step(a.action)
            buffer.add_batch(trajectory.from_transition(t, a, n))
            t = n
        rewards.append(float(t.reward))
        questions.append(len(env_py.asked_questions))

        # ② 파라미터 업데이트
        for _ in range(steps_per_ep):
            exp, _ = next(iterator)
            agent.train(exp)

        # ③ 로그
        if ep % 100 == 0:
            avg_r = np.mean(rewards[-100:])
            avg_q = np.mean(questions[-100:])
            print(f"[Ep {ep}] AvgReward={avg_r:.1f}  AvgQ={avg_q:.2f}")
            with summary_writer.as_default():
                tf.summary.scalar("AvgReward", avg_r, step=ep)

    # ─── 학습 종료: 그래프 & 모델 저장 ───
    plt.figure()
    plt.plot(rewards, label="보상"); plt.plot(questions, label="질문 수")
    plt.legend(); plt.tight_layout()
    plt.savefig("ai/log/training_plot.png", dpi=150); plt.close()

    spec = tf.TensorSpec([None, env_py.num_questions], tf.float32)

    @tf.function(input_signature=[spec])
    def serving_fn(inputs):
        q = q_net(inputs)
        if isinstance(q, (list, tuple)):
            q = q[0]          # 튜플이면 첫 Tensor만 추출
        return {"q_values": q}

    save_dir = "ai/models/q_net_saved"
    if os.path.exists(save_dir):
        import shutil, tempfile
        shutil.rmtree(save_dir)

    tf.saved_model.save(
        q_net,
        save_dir,
        signatures={"serving_default": serving_fn},
    )
    print("✅ SavedModel 저장 완료 :", save_dir)
    
# ── 실행 ─────────────────────────────────────────────
if __name__ == "__main__":
    automated_training(num_episodes=100, steps_per_ep=30)   # ≈ 10분 안쪽
