import random
import numpy as np
import json

class TwentyQuestionsEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_questions = len(dataset[0]['questions'])
        self.reset()

    def reset(self):
        self.target = random.choice(self.dataset)
        self.asked_questions = set()
        self.history = [-1] * self.num_questions
        self.done = False
        return self._get_state()
    
    def step(self, question_index):
        if self.done:
            raise ValueError("이미 게임이 종료되었습니다.")
        if question_index in self.asked_questions:
            return self._get_state(), -5, False
        
        self.asked_questions.add(question_index)
        answer = self.target['questions'][question_index]['answer']
        self.history[question_index] = answer

        reward = -1
        done = False

        if len(self.asked_questions) == self.num_questions:
            done = True
            self.done = True
        
        return self._get_state(), reward, done
    
    def guess(self, guess_name):
        if self.done:
            raise ValueError("이미 게임이 종료되었습니다.")
        
        self.done = True
        if guess_name == self.target['name']:
            return self._get_state(), 100, True
        else:
            return self._get_state(), -50, True
        
    def _get_state(self):
        return np.array(self.history, dtype=np.float32)
    
with open('ai/dataset/final_full_corrected_animal_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    animal_list = data['animals']

env = TwentyQuestionsEnv(animal_list)
num_episodes = 5

for episode in range(num_episodes):
    print(f"\n === 에피소드 {episode + 1} 시작 ===")
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        available_questions = list(set(range(env.num_questions)) - env.asked_questions)

        if available_questions:
            action = random.choice(available_questions)
            state, reward, done = env.step(action)
            total_reward += reward
            print(f"질문 {action+1}번: 보상 {reward}, 현재 상태 : {state}")
        else:
            break

    
guess_animal = random.choice([animal['name'] for animal in animal_list])
state, reward, done = env.guess(guess_animal)
total_reward += reward
print(f"동물 추축 : {guess_animal}, 추측 보상 : {reward}")
print(f"총 보상 : {total_reward}")
print(f"정답 동물 : {env.target['name']}")