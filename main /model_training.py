import os
import random
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import datetime

# Load and preprocess data
data = pd.read_csv("balanced_soil_dataset.csv")

def convert_depth(value):
    if isinstance(value, str) and '-' in value:
        parts = value.replace("cm", "").split("-")
        return (int(parts[0]) + int(parts[1])) / 2
    try:
        return float(value)
    except:
        return np.nan

data["depth"] = data["depth"].apply(convert_depth)
data = data.dropna()
X = data.drop("habitat", axis=1).values
y = data["habitat"].map({"not_habitable": 0, "moderate": 1, "habitat": 2}).values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# DQN with dueling architecture
def build_dueling_dqn(input_shape, num_actions):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)

    value = layers.Dense(64, activation="relu")(x)
    value = layers.Dense(1)(value)

    advantage = layers.Dense(64, activation="relu")(x)
    advantage = layers.Dense(num_actions)(advantage)

    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    model = models.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Experience Replay with Prioritized Experience Replay (simple version)
class PrioritizedReplayBuffer:
    def __init__(self, maxlen=100000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append((experience, abs(experience[2])))  # use reward as priority

    def sample(self, batch_size):
        priorities = np.array([p for (_, p) in self.buffer])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i][0] for i in indices]
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)

# Environment Simulator
class SoilEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0
        self.size = len(X)

    def reset(self):
        self.index = random.randint(0, self.size - 1)
        return self.X[self.index], self.y[self.index]

    def step(self, action):
        correct_label = self.y[self.index]
        reward = 1.0 if action == correct_label else -1.0
        done = True
        return self.X[self.index], reward, done, correct_label

# Training parameters
episodes = 10000
batch_size = 64
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.001

env = SoilEnv(X, y)
state_size = X.shape[1]
action_size = 3

model = build_dueling_dqn((state_size,), action_size)
target_model = build_dueling_dqn((state_size,), action_size)
target_model.set_weights(model.get_weights())

replay_buffer = PrioritizedReplayBuffer()

# TensorBoard setup
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Training loop
for ep in range(1, episodes + 1):
    state, true_label = env.reset()
    total_reward = 0
    action = np.argmax(model.predict(state.reshape(1, -1))[0]) if np.random.rand() > epsilon else random.randint(0, action_size - 1)

    next_state, reward, done, true_label = env.step(action)
    replay_buffer.add((state, action, reward, next_state, done))

    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        target_qs = target_model.predict(next_states)
        targets = model.predict(states)

        for i in range(batch_size):
            targets[i][actions[i]] = rewards[i] + (0.99 * np.max(target_qs[i]) * (1 - dones[i]))

        model.fit(states, targets, epochs=1, verbose=0, callbacks=[tensorboard_callback])

    total_reward += reward

    if epsilon > epsilon_min:
        epsilon -= epsilon_decay

    if ep % 10 == 0:
        target_model.set_weights(model.get_weights())

    if ep % 1000 == 0:
        model.save(f"alien_soil_model_ep{ep}.keras")
        print(f"\n✅ Saved model at episode {ep}")

    print(f"\U0001F680 Episode {ep} | ✅ Done | \U0001F3AF Action: {action} | \U0001F3F7️ True: {true_label} | \U0001F9E0 Epsilon: {epsilon:.4f}")
