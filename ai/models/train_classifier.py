# ai/models/train_classifier.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

# 데이터셋 로드
with open(os.path.join("ai", "dataset", "final_dataset_with_family_and_size.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
    animals = data["animals"]

features = []
labels = []
for animal in animals:
    feature = [q["answer"] for q in animal["questions"]]
    features.append(feature)
    labels.append(animal["name"])

features = np.array(features, dtype=np.float32)
unique_animals = sorted(list(set(labels)))
name_to_idx = {name: idx for idx, name in enumerate(unique_animals)}
int_labels = np.array([name_to_idx[name] for name in labels], dtype=np.int32)

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(features, int_labels, test_size=0.2, random_state=42)
num_classes = len(unique_animals)
input_dim = features.shape[1]

# MLP 분류 모델 구성
inference_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

inference_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
inference_model.summary()

history = inference_model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val))

# 모델 저장
model_dir = os.path.join("ai", "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "inference_model.h5")
inference_model.save(model_path)
print("분류 모델이 저장되었습니다:", model_path)
