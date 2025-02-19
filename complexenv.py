import numpy as np
import gymnasium as gym
from env import Source, Tank, Electrolyzer, Combustor, PowerOutput
from sourcefn import available_energy_complex, available_energy_simple

TANK_VOLUME = [12000, 12000]

CONVERSION_RATE = [0.5, 0.5]

ELECTROLYZERS_LOAD = [150, 150]

COMBUSTORS_LOAD = [150, 150]

SOURCE_MAX_POWER = 500
TARGET_POWER = 160

#politca 1: rescale, politica 2: clip


class ComplexEnv(gym.Env):
    
    def __init__(self):
        super(ComplexEnv, self).__init__()

        self.time = 0
        self.tank1 = Tank(TANK_VOLUME[0])
        self.tank2 = Tank(TANK_VOLUME[1])
        self.electrolyzer1 = Electrolyzer(CONVERSION_RATE[0], ELECTROLYZERS_LOAD[0])
        self.electrolyzer2 = Electrolyzer(CONVERSION_RATE[1], ELECTROLYZERS_LOAD[1])
        self.combustor1 = Combustor(CONVERSION_RATE[0], COMBUSTORS_LOAD[0])
        self.combustor2 = Combustor(CONVERSION_RATE[1], COMBUSTORS_LOAD[1])
        self.source = Source(SOURCE_MAX_POWER)
        self.output = PowerOutput()

        self.output_data = []
        self.volume_data1 = []
        self.volume_data2 = []
        self.reward_data = []
        self.action_data = []
        self.input_data = []

        # potenza in input, volume tanica 1, volume tanica 2
        obs_low = np.array([0, 0, 0])
        obs_high = np.array([1, 1, 1])

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape = (2,), dtype=np.float32)

        # percentuale corrente da convertire, percentuale idrogeno da convertire, 
        act_low = np.array([-1, -1, -1, -1])
        act_high = np.array([1, 1, 1, 1])
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, shape = (2,), dtype=np.float32)

    def _get_state(self):
        state = np.array(
            [self.source.get_source_power()/SOURCE_MAX_POWER, self.tank1.get_volume()/TANK_VOLUME[0], self.tank2.get_volume/TANK_VOLUME[1]], dtype=np.float32
        )
        return state
    
    def get_data(self):
        return self.reward_data, self.output_data, self.volume_data1, self.volume_data2, self.action_data, self.input_data

    def collect(self, reward, action) -> None:
        self.output_data.append(self.output.get_current_output())
        self.volume_data1.append(self.tank1.get_volume())
        self.volume_data2.append(self.tank2.get_volume())
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
        self.volume_data1 = []
        self.volume_data2 = []
        return self._get_state(), {}   
    
    def step(self, action):
        action = (action + 1) / 2

        power = self.source.set_source_power(self.time)
        power_to_h2 = self.electrolyzer.produce_hydrogen(action[0]*power, self.tank)
        h2_to_power = self.combustor.produce_power(action[1]*self.tank.get_volume(), self.tank)
        power = power + h2_to_power - power_to_h2
        self.output.set_output(power, self.time)
        
        next_state = self._get_state()
        error = relative_error(TARGET_POWER, self.output.get_current_output())
        current_output = self.output.get_current_output()
        #bonus serve per disincentivare l'output troppo basso 
        bonus = -(TARGET_POWER - current_output)/5 if current_output < TARGET_POWER else 5
        prevstate = prevstate_reward(self.output.get_previous_output(), current_output)
        distance = reward_quadratic(error)
        reward = distance + bonus + prevstate
        #print(distance, bonus, prevstate)
        self.collect(reward, action)
        self.time += 1
        truncated = self.time > 2000
        return next_state, reward, False, truncated, {}

    def render(self):
        print("Time: ", self.time)
        print("Input: ", self.source.get_source_power())
        print("Volume: ", self.tank.get_volume())
        print("Output: ", self.output.get_current_output())
        print("\n")
