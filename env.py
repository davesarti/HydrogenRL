import numpy as np
import gymnasium as gym

class Tank:

    def __init__(self, volume):
        self.total_volume = volume
        self.current_volume = 0
    
    def fill(self, amount):
        if(self.current_volume + amount > self.total_volume):
            amount = self.total_volume - self.current_volume
            self.current_volume = self.total_volume
        else:
            self.current_volume += amount
        return amount
    
    def empty(self, amount):
        if(self.current_volume - amount < 0):
            amount = self.current_volume
            self.current_volume = 0
        else:
            self.current_volume -= amount
        return amount
    
    def get_current_volume(self):
        return self.current_volume
    
    def reset_current_volume(self):
        self.current_volume = 0

class Electrolyzer:

    def __init__(self, conversion):
        self.conversion = conversion #conversione potenza -> idrogeno
    
    def produce_hydrogen(self, power, tank): #ritorna la potenza convertita
        filled_amount = tank.fill(power*self.conversion)
        return filled_amount/self.conversion
    
class Combustor:

    def __init__(self, conversion):
        self.conversion = conversion #conversione idrogeno -> potenza
    
    def produce_power(self, amount, tank): #ritorna la potenza prodotta
        produced = tank.get_current_volume()*self.conversion*amount
        tank.empty(produced/self.conversion)
        return produced

class Source:

    def __init__(self, mean):
        self.mean = mean
        self.set_source_power()

    def get_source_power(self):
        return self.current_power

    def set_source_power(self):
        self.current_power = np.random.normal(self.mean, 30)
        return self.current_power

class PowerOutput:

    def __init__(self):
        self.reset_output()

    def reset_output(self):
        self.current_output = 0

    def set_output(self, value):
        self.current_output = value

    def get_current_output(self):
        return self.current_output

class NetworkEnv(gym.Env):
    
    def __init__(self):
        super(NetworkEnv, self).__init__()

        self.time = 0
        self.tank = Tank(1000)
        self.electrolyzer = Electrolyzer(0.7)
        self.combustor = Combustor(0.7)
        self.source = Source(100)
        self.output = PowerOutput()

        self.output_data = []
        self.error_data = []
        self.volume_data = []
        self.reward_data = []
        self.action_data = []
        self.input_data = []

        # potenza in input, in output, volume tanica, errore della potenza in uscita
        obs_low = np.array([0, 0, 0])
        obs_high = np.array([200, 1000, 1000]) #da normalizzare

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape = (3,), dtype=np.float32)

        # percentuale corrente da convertire, percentuale idrogeno da convertire
        act_low = np.array([0,0])
        act_high = np.array([1,1])
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, shape = (2,), dtype=np.float32)

    def _get_state(self):
        state = np.array(
            [self.source.get_source_power(), self.output.get_current_output(), self.tank.get_current_volume()], dtype=np.float32
        )
        return state
    
    def get_data(self):
        return self.reward_data, self.output_data, self.volume_data, self.action_data, self.input_data

    def collect(self, reward, action):
        self.output_data.append(self.output.get_current_output())
        self.volume_data.append(self.tank.get_current_volume())
        self.reward_data.append(reward)
        self.action_data.append(action)
        self.input_data.append(self.source.get_source_power())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.source.set_source_power()
        self.tank.reset_current_volume()
        self.output.set_output(self.source.get_source_power())
        self.output_data = []
        self.volume_data = []
        self.action_data = []
        self.input_data = []
        self.time = 0
        return self._get_state(), {}        
    
    def step(self, action):
        self.time += 1
        power = self.source.set_source_power()
        power -= self.electrolyzer.produce_hydrogen(action[0]*power, self.tank)
        power += self.combustor.produce_power(action[1], self.tank)
        self.output.set_output(power)
        state = self._get_state()
        #reward = float(-10*(abs(state[1] - 100)/100)) # errore relativo con la media della potenza in input
        error = abs(state[1] - 100) / 100
        reward = float(-10 * np.exp(error))
        self.collect(reward, action)

        truncated = self.time > 1000

        return state, reward, False, truncated, {}

    def render(self):
        print("Time: ", self.time)
        print("Input: ", self.source.get_source_power())
        print("Volume: ", self.tank.get_current_volume())
        print("Output: ", self.output.get_current_output())
        print("\n")
    

    

    

