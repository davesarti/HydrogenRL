from env import NetworkEnv
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console
import torch

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

class ActionNoiseCallback(BaseCallback):
    def __init__(self, initial_sigma, final_sigma, total_timesteps, verbose=0):
        super(ActionNoiseCallback, self).__init__(verbose)
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps # num_timesteps viene incrementato automaticamente da stable-baselines
        new_sigma = self.initial_sigma + progress * (self.final_sigma - self.initial_sigma)
        self.model.action_noise.sigma = new_sigma
        return True

env = NetworkEnv()

check_env(env, warn=True)

n_actions = env.action_space.shape[-1]
initial_sigma = 0.5
final_sigma = 0.05
total_timesteps=100000
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=initial_sigma * np.ones(n_actions))

model = TD3(
    "MlpPolicy", 
    env, 
    action_noise=action_noise,
    verbose=1,
    device="cpu",
    learning_rate=5e-4,
    buffer_size=30000,
    batch_size=128,
    learning_starts=10000,
)

model.tau = 0.002

action_noise_callback = ActionNoiseCallback(initial_sigma, final_sigma, total_timesteps)

model.learn(total_timesteps,
        log_interval = 10,
        progress_bar = RichProgressBar(),
        callback=action_noise_callback
        )

rewards, outputs, volumes, actions, inputs = env.get_data()

model.save("hydrogen_TD3")

window_size = 100
smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
           
plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Reward per step', alpha=0.3)
plt.plot(smoothed_rewards, label='Trend reward (media mobile)', color='red')
plt.xlabel('Numero di step')
plt.ylabel('Reward')
plt.title('Tendenza generale della Reward')
plt.legend()

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