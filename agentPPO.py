from env import NetworkEnv, TARGET_POWER
from sourcefn import function_complex
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console

# Creazione di una progress bar personalizzata
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

# Creazione environment, monitoraggio e callback per modello migliore
env = NetworkEnv(function_complex)
val_env = Monitor(NetworkEnv(function_complex))

eval_callback = EvalCallback(
    val_env,
    best_model_save_path='./PPO/',
    eval_freq= 100000,
    deterministic=True,
    render=False
)

check_env(env, warn=True)

# Creazione del modello
model = PPO("MlpPolicy", env, n_epochs = 15, n_steps = 1024, batch_size = 128, gamma = 0.995, ent_coef = 0.01, learning_rate = 1e-4, device = "cpu")

# Caricamento del modello migliore
old_model = PPO.load("./PPO/best_model", env, device = "cpu")
model.policy.load_state_dict(old_model.policy.state_dict())

model.learn(total_timesteps = 1000000, log_interval = 10, progress_bar = RichProgressBar(), callback = eval_callback)

rewards, outputs, volumes, actions, inputs = env.get_data()

# Grafici su reward e volume
window = 50
smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
target = np.ones(len(outputs)) * TARGET_POWER

plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Reward per step', alpha=0.3)
plt.plot(smoothed_rewards, label='Trend reward (sliding window)', color='red')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Reward trend')
plt.legend()
plt.savefig("plots/reward_training.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(volumes)
plt.xlabel('Steps')
plt.ylabel('Energy storage volume')
plt.title("Volume trend")
plt.savefig("plots/volume_training.png", dpi=300, bbox_inches='tight')

plt.show()
env.close()