# https://keras.io/examples/rl/actor_critic_cartpole/
# import os

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000

# Создаем среду
env = gym.make("CartPole-v1")
env.reset(seed=seed)
eps = np.finfo(np.float32).eps.item()  # Предотвращение деления на ноль

num_inputs = 4
num_actions = 2
num_hidden = 128

# --- Архитектура модели на чистом TF 2.15 ---
inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = Model(inputs=inputs, outputs=[action, critic])

# Настройка оптимизатора и функции потерь из tensorflow.keras
optimizer = optimizers.Adam(learning_rate=0.01)
huber_loss = losses.Huber()

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

# while True:  # Обучаем, пока задача не будет решена
for i in range(0, 105):
    obs, _ = env.reset()
    episode_reward = 0

    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):

            state = tf.convert_to_tensor(obs, dtype=tf.float32)
            state = tf.expand_dims(state, 0)

            # Предсказание вероятностей действий и оценки критика
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Выбор действия на основе предсказанных вероятностей
            action_probs_np = action_probs.numpy()  # Конвертируем в numpy для np.random.choice
            action = np.random.choice(num_actions, p=np.squeeze(action_probs_np))

            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Шаг в среде
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            done = terminated or truncated
            if done:
                break

        # Считаем скользящую среднюю награду
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Расчет дисконтированных наград (Labels для Критика)
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Нормализация наград
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Расчет лоссов для Актёра и Критика
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # лосс актера

            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Обратное распространение ошибки (Backpropagation)
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Очистка истории перед новым эпизодом
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Логирование
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    # Условие победы (в CartPole-v1 удержание палочки в течение 195+ шагов означает успех)
    if running_reward > 195:
        print("Solved at episode {}!".format(episode_count))
        break
