from env import NetworkEnv, TARGET_POWER
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from stable_baselines3.common.noise import NormalActionNoise

env = NetworkEnv()
model = PPO.load("./PPO/best_model_complex2", env)
check_env(env, warn=True)

obs, _ = env.reset()  # Estrai solo l'osservazione dalla tupla
truncated = False
output_val_data = []
input_val_data = []
while not truncated:
    input_val_data.append(env.source.get_source_power())
    action, _states = model.predict(obs, deterministic=True)
    next_state, reward, done, truncated, _ = env.step(action)
    output_val_data.append(env.output.get_current_output())
    obs = next_state

target = np.ones(len(output_val_data)) * TARGET_POWER
input_avg = np.ones(len(input_val_data))*np.mean(input_val_data)

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(input_val_data, label="Input")
plt.plot(output_val_data, label="Output")
plt.plot(target, label="Target")
plt.plot(input_avg, label="Input avg", linestyle='--', color = "gray")
plt.xlabel("Numero di step")
plt.ylabel("Potenza")
plt.title("Outputs")
plt.legend()
plt.show()

env.close()