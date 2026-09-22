import numpy as np
import gymnasium as gym
from .sourcemodel import Net
import torch
from .paths import SOURCE_MODEL_PATH

CONVERSION_RATE = 0.7
TANK_VOLUME = 750000
SOURCE_MAX_POWER = 4000
TARGET_POWER = 1300

model = Net()
model.load_state_dict(torch.load(SOURCE_MODEL_PATH, map_location="cpu"))
model.eval()


def validate_percentage(value) -> None:
    if value < 0 or value > 1:
        print('Percentages must be between 0 and 1')

def reward_quadratic(error, scale = 25):
    return -scale * (error ** 1.5) + 5

def prevstate_reward(prevalue, value):
    if prevalue == 0:
        return 0
    error = relative_error(prevalue, value)
    return -100*error

def relative_error(target, value):
    return abs(target - value) / target

class Source:

    def __init__(self, source_function, maximum: float, time: int = 0) -> None:
        self.maximum = maximum
        self.source_function = source_function
        self.set_source_power(time)
        
    def get_source_power(self) -> float:
        return self.current_power
    
    # Set source power based on time
    def set_source_power(self, time: int) -> float:
        self.current_power = model(torch.tensor(self.source_function(time), dtype=torch.float32).reshape(-1,1)).item() 
        return self.current_power

class Tank:

    def __init__(self, volume: float) -> None:
        self.total_volume = volume  # absolute value
        self.volume = 0
    
    # Fill the tank with a given amount and return the amount actually filled
    def fill(self, amount: float) -> float:
        if(self.volume + amount > self.get_total_volume()):
            amount = self.get_total_volume() - self.volume
            self.volume = self.get_total_volume()
        else:
            self.volume += amount
        return amount
    
    # Empty the tank by a given amount and return the amount actually emptied
    def empty(self, amount: float) -> float:
        if(self.volume - amount < 0):
            amount = self.volume
            self.volume = 0
        else:
            self.volume -= amount
        return amount
    
    def get_volume(self) -> float:
        return self.volume
    
    def get_total_volume(self) -> float:
        return self.total_volume
    
    def reset_volume(self) -> None:
        self.volume = 0

class Electrolyzer:

    def __init__(self, conversion: float) -> None:
        validate_percentage(conversion)
        self.conversion = conversion  # power -> hydrogen conversion
    
    # Given the input power and a tank, return the energy equivalent to the hydrogen actually produced
    def produce_hydrogen(self, power: float, tank: Tank) -> float:
        produced = power * self.conversion
        filled_amount = tank.fill(produced)
        return filled_amount/self.conversion
    
class Combustor:

    def __init__(self, conversion: float) -> None:
        validate_percentage(conversion)
        self.conversion = conversion  # hydrogen -> power conversion
    
    # Given the amount of hydrogen and a tank, return the power actually produced
    def produce_power(self, amount: float, tank: Tank) -> float:  # returns the power produced
        available_volume = tank.empty(amount)
        real_power = available_volume * self.conversion
        return real_power

class PowerOutput:

    def __init__(self) -> None:
        self.current_output = 0
        self.previous_output = 0

    def set_output(self, value: float, time: int = 0) -> None:
        if(time == 0):
            self.current_output = value
            self.previous_output = value
        else:
            self.previous_output = self.current_output
            self.current_output = value

    def get_current_output(self) -> float:
        return self.current_output
    
    def get_previous_output(self) -> float:
        return self.previous_output
    
class NetworkEnv(gym.Env):
    
    def __init__(self, source_function, naive = False):
        super(NetworkEnv, self).__init__()

        self.time = 0
        self.tank = Tank(TANK_VOLUME)
        # maximum load is set to be ineffective
        self.electrolyzer = Electrolyzer(CONVERSION_RATE) 
        self.combustor = Combustor(CONVERSION_RATE)
        self.source = Source(source_function, SOURCE_MAX_POWER)
        self.output = PowerOutput()

        self.naive = naive  # If True, the model does not consider agent actions but a heuristic

        self.output_data = []
        self.reward_data = []
        self.action_data = []
        self.input_data = []
        self.volume_data = []

        # States defined by source power and tank volume
        obs_low = np.array([0, 0])
        obs_high = np.array([1, 1])

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape = (2,), dtype=np.float32)

        # Actions defined by current to convert and percentage of hydrogen to convert
        act_low = np.array([-1, -1])
        act_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, shape = (2,), dtype=np.float32)

    def _get_state(self):
        state = np.array(
            [self.source.get_source_power()/SOURCE_MAX_POWER, self.tank.get_volume()/TANK_VOLUME], dtype=np.float32
        )
        return state
    
    def get_data(self):
        return self.reward_data, self.output_data, self.volume_data, self.action_data, self.input_data

    def collect(self, reward, action) -> None:
        self.output_data.append(self.output.get_current_output())
        self.volume_data.append(self.tank.get_volume())
        self.reward_data.append(reward)
        self.action_data.append(action)
        self.input_data.append(self.source.get_source_power())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.source.set_source_power(0)
        self.tank.reset_volume()
        self.output.set_output(0)
        self.output_data = []
        self.action_data = []
        self.input_data = []
        return self._get_state(), {}   
    
    def step(self, action):
        # Actions are normalized between -1 and 1, they are transformed from -1 to 1 into 0 to 1
        action = (action + 1) / 2

        if(self.naive):
            source_power = self.source.set_source_power(self.time)
            if(source_power > TARGET_POWER):
                h2_to_power = 0
                power_to_h2 = self.electrolyzer.produce_hydrogen(source_power - TARGET_POWER, self.tank)
            elif(source_power < TARGET_POWER):
                power_to_h2 = 0
                h2_to_power = self.combustor.produce_power((TARGET_POWER - source_power)/0.7, self.tank)
        
        else:
            source_power = self.source.set_source_power(self.time)
            h2_to_power = self.combustor.produce_power(action[1]*self.tank.get_volume(), self.tank)
            power_to_h2 = self.electrolyzer.produce_hydrogen(action[0]*source_power, self.tank)

        power = source_power + h2_to_power - power_to_h2
        self.output.set_output(power, self.time)
        
        next_state = self._get_state()
        current_output = self.output.get_current_output()

        error = relative_error(TARGET_POWER, current_output)        
        distance = reward_quadratic(error)
        bonus = -abs(TARGET_POWER - current_output)/50 if TARGET_POWER > current_output else 10
        storage_bonus = power_to_h2 * 0.01 - h2_to_power * 0.02 if source_power > TARGET_POWER * 1.1 else 0
        prevstate = prevstate_reward(self.output.get_previous_output(), current_output)

        reward = float(distance + prevstate + bonus + storage_bonus)
        reward = np.clip(reward, -30, None)

        self.collect(reward, action)
        truncated = self.time >= 10000
        self.time += 1
        return next_state, reward, False, truncated, {}

    def render(self):
        print("Time: ", self.time)
        print("Input: ", self.source.get_source_power())
        print("Volume: ", self.tank.get_volume())
        print("Output: ", self.output.get_current_output())
        print("\n")