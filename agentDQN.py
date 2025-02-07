from env import NetworkEnv
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cpu")

class ReplayMemory(object): #implementazione replay memory per experience replay

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
env = NetworkEnv()
n_actions = env.action_space.shape[0]
n_observations = env.observation_space.shape[0]

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
num_episodes = 50

def select_action(state): #implementazione politica epsilon greedy
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            print("bestvalue")
            output = policy_net(state)
            print(output)
            return output.max(1).indices.view(1, 1) #selezione azione con value migliore
    else:
        print("random")
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.float32)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # This converts batch-array of Transitions to Transition of batch-arrays
    # ovvero da [(s1, a1, ns1, r1), (s2, a2, ns2, r2)] a [(s1,s2), (a1,a1), (ns1, ns2), (r1,r2)]
    batch = Transition(*zip(*transitions)) 

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    #otteniamo un tensore con i Q-value delle azioni selezionate per ogni transizione nel batch
    state_action_values = policy_net(state_batch).gather(1, action_batch) 

    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    #unsqueeze aggiunge una dimensione per rendere le shape compatibili
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) 

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False

    while not done:
        action = select_action(state)
        print(action)
        observation, reward, done, _ = env.step(action[0].tolist())     
        
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        if done:
            break

    target_net.load_state_dict(policy_net.state_dict())

print('Complete')