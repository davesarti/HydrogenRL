import numpy as np

TARGET_POWER = 1000

def prevstate_reward(prevalue, value):
    if prevalue == 0:
        return 0
    error = relative_error(prevalue, value)
    return 15*np.tanh(-error)

def relative_error(target, value):
    return abs(target - value) / target

def reward_quadratic(error, scale = 8):
    return -scale * (error ** 1.5)

x = np.linspace(0, 4000, 4000)
rewards = []

for value in x:
    #bonus = -(TARGET_POWER - value)/40 if value < TARGET_POWER else 15
    error = relative_error(TARGET_POWER, value)
    if TARGET_POWER > value:
        bonus = -abs(TARGET_POWER - value)/30
    else:
        bonus = 15
    distance = reward_quadratic(error)
    reward = bonus + distance
    rewards.append(reward)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(x, rewards, label='Reward per step', alpha=0.3)
plt.xlabel('Potenza')
plt.ylabel('Reward')
plt.title('Reward in funzione della potenza')
plt.legend()

plt.show()

