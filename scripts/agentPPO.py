from hydrogen_rl.env import NetworkEnv, TARGET_POWER
from hydrogen_rl.paths import MODEL_DIR, PLOT_DIR
from hydrogen_rl.sourcefn import function_complex
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console

# Create a custom progress bar
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

# Create environment, monitoring, and callback for best model
env = NetworkEnv(function_complex)
val_env = Monitor(NetworkEnv(function_complex))

eval_callback = EvalCallback(
    val_env,
    best_model_save_path=str(MODEL_DIR),
    eval_freq= 100000,
    deterministic=True,
    render=False
)

check_env(env, warn=True)

# Create the model
model = PPO("MlpPolicy", env, n_epochs = 15, n_steps = 1024, batch_size = 128, gamma = 0.995, ent_coef = 0.01, learning_rate = 1e-4, device = "cpu")

model.learn(total_timesteps = 1000000, log_interval = 10, progress_bar = RichProgressBar(), callback = eval_callback)

rewards, outputs, volumes, actions, inputs = env.get_data()

# Plots for reward and volume
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
plt.savefig(PLOT_DIR / "reward_training.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(volumes)
plt.xlabel('Steps')
plt.ylabel('Energy storage volume')
plt.title("Volume trend")
plt.savefig(PLOT_DIR / "volume_training.png", dpi=300, bbox_inches='tight')

plt.show()
env.close()