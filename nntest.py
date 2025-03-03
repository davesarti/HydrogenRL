import torch
from sourcefn import available_energy_complex, available_energy_simple
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sourcemodel import Net
import numpy as np

model = Net()
model.load_state_dict(torch.load("sourcefn_model.pth"))
model.eval()

x_data = pd.read_csv("csv/wind.csv")
y_predict = model(torch.tensor(x_data.values).float())
y_predict = y_predict.detach().numpy()
y_data = pd.read_csv("csv/power.csv")

plt.figure(figsize=(12, 6))
plt.scatter(x_data.values, y_data.values, color='green', label='Actual Function')
plt.scatter(x_data.values, y_predict, color='blue', label='Predicted Function')
plt.xlabel('Wind speed')
plt.ylabel('Power')
plt.legend()

plt.show()

complex_data = []
simple_data = []

for i in range (0,10000):
    complex_energy = available_energy_complex(i, 12)
    simple_energy = available_energy_simple(i, 24)
    complex_data.append(complex_energy)
    simple_data.append(simple_energy)

complex_nn = model(torch.tensor(complex_data).float().reshape(-1,1))
simple_nn = model(torch.tensor(simple_data).float().reshape(-1,1))
t = np.linspace(0, 10000, 10000)

plt.figure(figsize=(12, 6))
plt.plot(t, complex_data, label='Complex input')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(t, complex_nn.detach().numpy(), label= 'Complex output')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(t, simple_data, label='Input')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(t, simple_nn.detach().numpy(), label='Output')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()

plt.show()