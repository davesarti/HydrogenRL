from env import NetworkEnv
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from stable_baselines3.common.noise import NormalActionNoise

env = NetworkEnv()
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model = TD3.load("hydrogen_TD3", env, print_system_info=True)
check_env(env, warn=True)

obs, _ = env.reset()  # Estrai solo l'osservazione dalla tupla
truncated = False
output_val_data = []
while not truncated:
    action, _states = model.predict(obs, deterministic=True)
    next_state, reward, done, truncated, _ = env.step(action)
    output_val_data.append(env.output.get_current_output())
    obs = next_state

plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(output_val_data)
plt.title("Outputs")
plt.show()

env.close()

