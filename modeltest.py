from env import NetworkEnv
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from stable_baselines3.common.noise import NormalActionNoise

env = NetworkEnv()
model = PPO.load("./PPO/best_model_fix", env)
check_env(env, warn=True)

obs, _ = env.reset()  # Estrai solo l'osservazione dalla tupla
truncated = False
output_val_data = []
while not truncated:
    action, _states = model.predict(obs, deterministic=True)
    next_state, reward, done, truncated, _ = env.step(action)
    output_val_data.append(env.output.get_output())
    obs = next_state

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(output_val_data)
plt.xlabel("Numero di step")
plt.ylabel("Output")
plt.title("Outputs")
plt.show()

env.close()