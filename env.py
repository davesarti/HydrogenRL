import numpy as np
import gymnasium as gym

def validate_percentage(value: float) -> None:
    if value < 0 or value > 1:
        print('Le percentuali sono tra 0 e 1')

def reward_quadratic(error, k=10):
    return float(100 / (1 + k * error**2))

def reward_logistic(error, k=10):
    return 100 / (1 + np.exp(k * (error - 0.5)))

def reward_exponential_penalty(error):
    return 100 * (1 - np.exp(error))

def reward_bounded_exp_PPO(error, k=2):
    return float(20 * np.exp(-k * error) - 10)

class Source:

    def __init__(self, maximum: float, time: int = 0):
        self.maximum = maximum
        self.set_source_power(time)

    def get_source_power(self):
        return self.current_power

    def set_source_power(self, time: int):
        self.current_power = ((np.sin(time)+1)/2)*self.maximum
        return self.current_power


class Tank:

    def __init__(self, volume: float) -> None:
        self.total_volume = volume #assoluto
        self.volume = 0
    
    # Riempie la tanica di una data quantità e ritorna la quantità riempita
    def fill(self, amount: float) -> float:
        if(self.volume + amount > self.get_total_volume()):
            amount = self.get_total_volume() - self.volume
            self.volume = self.get_total_volume()
        else:
            self.volume += amount
        return amount
    
    # Svuola la tanica di una data quantità e ritorna la quantità svuotata
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

    def __init__(self, conversion: float):
        validate_percentage(conversion)
        self.conversion = conversion #conversione potenza -> idrogeno
    
    # Produce una data quantità di idrogeno data la corrente in entrata e una tanica, ritorna l'energia equivalente all'idrogeno prodotto
    def produce_hydrogen(self, power: float, tank: Tank) -> float:
        produced = power * self.conversion
        filled_amount = tank.fill(produced)
        return filled_amount/self.conversion
    
class Combustor:

    def __init__(self, conversion: float):
        validate_percentage(conversion)
        self.conversion = conversion #conversione idrogeno -> potenza
    
    # Data la quantità di idrogeno e una tanica, ritorna la corrente effettivamente prodotta
    def produce_power(self, amount: float, tank: Tank) -> float: #ritorna la potenza prodotta
        available_volume = tank.empty(amount)
        real_power = available_volume * self.conversion
        return real_power

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
        self.source = Source(300)
        self.output = PowerOutput()

        self.output_data = []
        self.error_data = []
        self.volume_data = []
        self.reward_data = []
        self.action_data = []
        self.input_data = []

        # potenza in input, in output, volume tanica
        obs_low = np.array([0, 0, 0])
        obs_high = np.array([500, 500, 500])

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape = (3,), dtype=np.float32)

        # percentuale corrente da convertire, percentuale idrogeno da convertire
        act_low = np.array([-1, -1])
        act_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, shape = (2,), dtype=np.float32)

    def _get_state(self):
        state = np.array(
            [self.source.get_source_power(), self.output.get_current_output(), self.tank.get_volume()], dtype=np.float32
        )
        return state
    
    def get_data(self):
        return self.reward_data, self.output_data, self.volume_data, self.action_data, self.input_data

    def collect(self, reward, action):
        self.output_data.append(self.output.get_current_output())
        self.volume_data.append(self.tank.get_volume())
        self.reward_data.append(reward)
        self.action_data.append(action)
        self.input_data.append(self.source.get_source_power())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.source.set_source_power(0)
        self.tank.reset_volume()
        self.output.set_output(self.source.get_source_power())
        self.output_data = []
        self.volume_data = []
        self.action_data = []
        self.input_data = []
        self.time = 0
        return self._get_state(), {}   
    
    def step(self, action):
        action = (action + 1) / 2

        self.time += 1
        power = self.source.set_source_power(self.time)
        power = power - self.electrolyzer.produce_hydrogen(action[0]*power, self.tank)
        power = power + self.combustor.produce_power(action[1]*self.tank.get_volume(), self.tank)
        self.output.set_output(power)
        
        state = self._get_state()
        error = abs(state[1] - 100) / 100
        reward = reward_bounded_exp_PPO(error)
        self.collect(reward, action)
        truncated = self.time > 1000
        return state, reward, False, truncated, {}

    def render(self):
        print("Time: ", self.time)
        print("Input: ", self.source.get_source_power())
        print("Volume: ", self.tank.get_volume())
        print("Output: ", self.output.get_current_output())
        print("\n")






