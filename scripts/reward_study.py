import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from hydrogen_rl.env import relative_error, reward_quadratic
from hydrogen_rl.paths import PLOT_DIR

def prevstate_reward(prev_error):
    return -100*prev_error

TARGET_POWER = 1300
power = np.linspace(0, 4000, 4000)
prevstate = np.linspace(0, 1, 4000)

rewards = np.zeros((len(power), len(prevstate)))

for i in range(len(prevstate)):
    for j in range(len(power)):
        value = power[i]
        prev_error = prevstate[j]
        pow_error = relative_error(TARGET_POWER, value)
        if TARGET_POWER > value:
            bonus = -abs(TARGET_POWER - value)/90
        else:
            bonus = 5
        distance = reward_quadratic(pow_error)
        prev_reward = prevstate_reward(prev_error)
        reward = bonus + distance + prev_reward
        reward = np.clip(reward, -20, None)
        rewards[i][j] = reward

plt.figure(figsize=(12, 6))

colors = [(0.8, 0, 0), (1, 1, 1), (0, 0.8, 0)] 
cmap = LinearSegmentedColormap.from_list("reward_cmap", colors, N=100)

im = plt.imshow(rewards, cmap=cmap, aspect='auto', 
                extent=[prevstate.min(), prevstate.max(), power.min(), power.max()],
                origin='lower', interpolation='bilinear')

cbar = plt.colorbar(im)
cbar.set_label('Reward')

plt.ylabel('Power (kW)')
plt.xlabel('Error with previous output')
plt.title('Reward Heatmap')

plt.axhline(y=TARGET_POWER, color='black', linestyle='--', linewidth=1, alpha=0.7, 
            label=f'Target Power ({TARGET_POWER} kW)')

plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "reward_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()