import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

with open("ai/dataset/common_sense_dataset.json", "r", encoding="utf-8") as f:
    data  = json.load(f)

question_keys = list(data[0]["questions"].keys())
X = []
y = []

for item in data:
    answers = [item["questions"][q] for q in question_keys]
    X.append(answers)
    y.append(item["name"])

X = np.array(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

sequence_length = len(question_keys)
vocab_size = 3
num_characters = len(label_encoder.classes_)

model = Sequential([
    Embedding(input_dim=vocab_size + 1, output_dim=128, input_length=sequence_length),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.3),  # 과적합 방지
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_characters, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y_encoded, epochs=100, batch_size = 8)

# 모델 저장
model.save("ai/models/model.h5")

# 라벨 인코더 저장
import pickle
with open("ai/models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)