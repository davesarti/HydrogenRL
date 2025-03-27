from env import NetworkEnv, TARGET_POWER
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt

STEPS = 10000

# Caricamento dell'ambiente e del modello
env = NetworkEnv()
model = PPO.load("./PPO/best_model_best", env)
check_env(env, warn=True)

# Esecuzione del modello
obs, _ = env.reset()  
truncated = False

for i in range(STEPS):
    action, _states = model.predict(obs, deterministic=True)
    next_state, reward, done, truncated, _ = env.step(action)
    obs = next_state

reward_data, output_data, volume_data, action_data, input_data = env.get_data()

# Grafico dei dati di input, output e target
target = np.ones(len(output_data)) * TARGET_POWER
output_avg = np.ones(len(output_data))*np.mean(output_data)

plt.figure(figsize=(20, 20))
plt.plot(input_data, label="Input")
plt.plot(output_data, label="Output")
plt.plot(target, label="Target")
plt.plot(output_avg, label="Output avg", linestyle='--', color = "gray")
plt.xlabel("Steps")
plt.ylabel("Power")
plt.title("Input and Output")
plt.legend()
plt.savefig("plots/input_output.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(volume_data)
plt.xlabel('Steps')
plt.ylabel('Volume')
plt.title("Energy storage volume")
plt.savefig("plots/volume.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(action_data)
plt.xlabel('Steps')
plt.ylabel('Action')
plt.title("Actions")
plt.legend(["Pow to H2", "H2 to Pow"])
plt.savefig("plots/actions.png", dpi=300, bbox_inches='tight')

plt.show()

env.close()