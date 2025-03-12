from env import NetworkEnv

for episode in range(10):
    env = NetworkEnv()
    state = env.reset()
    truncated = False
    total_reward = 0
    while not truncated:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        print(f"Action: {action} \nReward: {reward}")
        env.render()
    print(f"Episode {episode + 1}: Total reward: {total_reward}")

env.close()
