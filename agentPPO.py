from env import NetworkEnv
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console


class RichProgressBar:
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.elapsed] Time elapsed: {task.elapsed}"),
            TextColumn("[progress.remaining] Time remaining: {task.remaining}"),
            console=self.console,
        )
        self.task = None

env = NetworkEnv()

check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000, log_interval = 10, progress_bar = RichProgressBar())

rewards, outputs, volumes, actions, inputs = env.get_data()

window_size = 100
smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
smoothed_outputs = np.convolve(outputs, np.ones(window_size)/window_size, mode='valid')
smoothed_inputs = np.convolve(inputs, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(14, 10))
           
plt.subplot(3, 1, 1)
plt.plot(smoothed_rewards)
plt.title("Rewards")

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(volumes)
plt.title("Volumes")

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(actions)
plt.title("Actions")
plt.legend(["Pow to H2", "H2 to Pow"])

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(inputs, label="Inputs")
plt.plot(outputs, label="Outputs")
plt.title("Inputs and Outputs")
plt.legend()

plt.show()
env.close()

#TODO: riprovare con PPO (ricordati clipping) con plot step-action da mandare a mario, cercare reward più stabile per DDPG