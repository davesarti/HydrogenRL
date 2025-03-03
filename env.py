import numpy as np
import gymnasium as gym
from sourcefn import available_energy_complex, available_energy_simple, wind_data
from sourcemodel import Net
import torch

#media dei dati di potenza = 1300

CONVERSION_RATE = 0.8
TANK_VOLUME = 60000
SOURCE_MAX_POWER = 4000
TARGET_POWER = 800
MAX_WIND = 20

model = Net()
model.load_state_dict(torch.load("sourcefn_model.pth"))
model.eval()
#Tarare correttamente questi parametri è cruciale perchè influenzano la convergenza dell'algoritmo
#Pare che una buona approssimazione della dipendenza dei parametri sia TARGET_POWER < SOURCE_MAX_POWER/2 * CONVERSION_RATE**2

def validate_percentage(value) -> None:
    if value < 0 or value > 1:
        print('Le percentuali sono tra 0 e 1')

def reward_quadratic(error, scale = 1.5, min_reward = -50):
    r = -scale * (error ** 2)
    return np.clip(r, min_reward, None)

def prevstate_reward(prevalue, value):
    if prevalue == 0:
        return 0
    error = relative_error(prevalue, value)
    return 6*np.tanh(-error)

def relative_error(target, value):
    return abs(target - value) / target

class Source:

    def __init__(self, maximum: float, time: int = 0) -> None:
        self.maximum = maximum
        self.set_source_power(time)

    def get_source_power(self) -> float:
        return self.current_power
    
    # Setta la potenza della sorgente in base al tempo
    def set_source_power(self, time: int) -> float:
        self.current_power = model(torch.tensor(wind_data(time)).reshape(-1,1)).item()
        return self.current_power

class Tank:

    def __init__(self, volume: float) -> None:
        self.total_volume = volume #assoluto
        self.volume = 20000
    
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

    def __init__(self, conversion: float, load : float) -> None:
        validate_percentage(conversion)
        self.conversion = conversion #conversione potenza -> idrogeno
        self.load = load
    
    # Data la corrente in entrata e una tanica, ritorna l'energia equivalente all'idrogeno effettivamente prodotto
    def produce_hydrogen(self, power: float, tank: Tank) -> float:
        if(power > self.load):
            power = self.load
        produced = power * self.conversion
        filled_amount = tank.fill(produced)
        return filled_amount/self.conversion
    
class Combustor:

    def __init__(self, conversion: float, load: float) -> None:
        validate_percentage(conversion)
        self.conversion = conversion #conversione idrogeno -> potenza
        self.load = load
    
    # Data la quantità di idrogeno e una tanica, ritorna la corrente effettivamente prodotta
    def produce_power(self, amount: float, tank: Tank) -> float: #ritorna la potenza prodotta
        if(amount > self.load):
            amount = self.load
        available_volume = tank.empty(amount)
        real_power = available_volume * self.conversion
        return real_power

class PowerOutput:

    def __init__(self) -> None:
        self.current_output = 0
        self.previous_output = 0

    def set_output(self, value: float, time: int = 0) -> None:
        if(time == 0):
            self.previous_output = self.current_output
        self.previous_output = self.current_output
        self.current_output = value

    def get_current_output(self) -> float:
        return self.current_output
    
    def get_previous_output(self) -> float:
        return self.previous_output
    
class NetworkEnv(gym.Env):
    
    def __init__(self):
        super(NetworkEnv, self).__init__()

        self.time = 0
        self.tank = Tank(TANK_VOLUME)
        #il carico massimo è settato in modo da esse ininfluente
        self.electrolyzer = Electrolyzer(CONVERSION_RATE, SOURCE_MAX_POWER) 
        self.combustor = Combustor(CONVERSION_RATE, TANK_VOLUME)
        self.source = Source(SOURCE_MAX_POWER)
        self.output = PowerOutput()

        self.output_data = []
        self.reward_data = []
        self.action_data = []
        self.input_data = []
        self.volume_data = []

        # potenza in input, volume tanica
        obs_low = np.array([0, 0])
        obs_high = np.array([1, 1])

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape = (2,), dtype=np.float32)

        # percentuale corrente da convertire, percentuale idrogeno da convertire
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
        action = (action + 1) / 2

        power = self.source.set_source_power(self.time)
        h2_to_power = self.combustor.produce_power(action[1]*self.tank.get_volume(), self.tank)
        power_to_h2 = self.electrolyzer.produce_hydrogen(action[0]*power, self.tank)
        power = power + h2_to_power - power_to_h2
        self.output.set_output(power, self.time)
        
        next_state = self._get_state()
        error = relative_error(TARGET_POWER, self.output.get_current_output())
        current_output = self.output.get_current_output()
        #bonus serve per disincentivare l'output troppo basso 
        bonus = -(TARGET_POWER - current_output)/20 if current_output < TARGET_POWER else 10
        prevstate = prevstate_reward(self.output.get_previous_output(), current_output)
        distance = reward_quadratic(error)
        reward = float(distance + bonus) # prevstate
        
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
