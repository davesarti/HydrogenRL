from env import NetworkEnv, TARGET_POWER
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
env = NetworkEnv()
val_env = Monitor(NetworkEnv())

eval_callback = EvalCallback(
    val_env,
    best_model_save_path='./PPO/',
    eval_freq= 100000,
    deterministic=True,
    render=False
)

check_env(env, warn=True)

# Caricamento/allenamento del modello
model = PPO("MlpPolicy", env, gamma = 0.99, device = "cpu")
#model = PPO.load("./PPO/best_model_hugeps", env, device = "cpu")
model.learn(total_timesteps = 1000000, log_interval = 10, progress_bar = RichProgressBar(), callback = eval_callback)

rewards, outputs, volumes, actions, inputs = env.get_data()

# Grafici su reward, azioni, input e output
window = 50
smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
target = np.ones(len(outputs)) * TARGET_POWER

plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Reward per step', alpha=0.3)
plt.plot(smoothed_rewards, label='Trend reward (media mobile)', color='red')
plt.xlabel('Numero di step')
plt.ylabel('Reward')
plt.title('Tendenza generale della Reward')
plt.legend()
plt.savefig("plots/reward.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(volumes)
plt.xlabel('Numero di step')
plt.ylabel('Volume')
plt.title("Volumes")
plt.savefig("plots/volume.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(actions)
plt.xlabel('Numero di step')
plt.ylabel('Azione')
plt.title("Actions")
plt.legend(["Pow to H2", "H2 to Pow"])
plt.savefig("plots/actions.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(inputs, label="Inputs")
plt.plot(outputs, label="Outputs")
plt.plot(target, label="Target")
plt.xlabel('Numero di step')
plt.ylabel('Potenza')
plt.title("Inputs and Outputs")
plt.legend()
plt.savefig("plots/inputs_outputs.png", dpi=300, bbox_inches='tight')

plt.show()
env.close()