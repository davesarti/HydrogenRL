from sourcefn import available_energy_complex
from sourcemodel import Net
import torch
import matplotlib.pyplot as plt
import numpy as np

fd = open("csv/wind.csv", "r")
wind = fd.readlines()
fd.close()
wind = [float(x.strip()) for x in wind[1:]]  # rimuove l'header e converte in float

fd = open("csv/wind_sv.csv", "r")
wind_sv = fd.readlines()
fd.close()
wind_sv= [float(x.strip()) for x in wind_sv[1:]]  # rimuove l'header e converte in float

x = [i for i in range(10000)]

plt.figure(figsize=(12, 6))
plt.plot(x, wind[:10000], label='Wind speed', color='blue')
plt.plot(x, wind_sv[:10000], label='Wind speed smoothed', color='red')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()

plt.show()

model = Net()
model.load_state_dict(torch.load("sourcefn_model.pth"))
model.eval()

power = []
power_sv = []
for i in range(10000):
    power.append(model(torch.tensor(wind[i]).reshape(-1,1)).item())
    power_sv.append(model(torch.tensor(wind_sv[i]).reshape(-1,1)).item())

print(power)

plt.figure(figsize=(12, 6))
plt.plot(x, power, label='Power given wind speed', color='blue')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(x, power_sv, label='Power given wind speed smoothed', color='red')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()

plt.show()