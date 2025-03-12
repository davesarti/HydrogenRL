import torch
from sourcefn import available_energy_complex
import pandas as pd
import matplotlib.pyplot as plt
from sourcemodel import Net
import numpy as np

# Caricamento del modello
model = Net()
model.load_state_dict(torch.load("sourcefn_model.pth"))
model.eval()

# Caricamento dei dati (non smoothed) e predizione
x_data = pd.read_csv("csv/wind.csv")
y_predict = model(torch.tensor(x_data.values).float())
y_predict = y_predict.detach().numpy()
y_data = pd.read_csv("csv/power.csv")

# Grafico della funzione di potenza generata in base alla velocità del vento
plt.figure(figsize=(12, 6))
plt.scatter(x_data.values, y_data.values, color='green', label='Actual Function')
plt.scatter(x_data.values, y_predict, color='blue', label='Predicted Function')
plt.xlabel('Wind speed')
plt.ylabel('Power')
plt.legend()

plt.show()

# Caricamento dei dati smoothed e funzione qualitativa per il vento
x_data = pd.read_csv("csv/wind_sv.csv")
real_data = x_data[1:10001].to_numpy()
sim_data = []
for i in range (0,10000):
    complex_energy = available_energy_complex(i/20, 10)
    sim_data.append(complex_energy)

# Predizione della potenza dato il vento
sim_nn = model(torch.tensor(sim_data).float().reshape(-1,1))
real_nn = model(torch.tensor(real_data).float().reshape(-1,1))
t = np.linspace(0, 10000, 10000)

# Confronto tra dati reali e simulati con relative predizioni
plt.figure(figsize=(12, 6))
plt.plot(t, real_data, label='Real data', color='blue')
plt.plot(t, sim_data, label='Simulated data', color='red')
plt.xlabel('Time')
plt.ylabel('Wind')
plt.legend()
plt.title('Wind smoothed vs simulated')

plt.figure(figsize=(12, 6))
plt.plot(t, real_nn.detach().numpy(), label='Real data', color='blue')
plt.plot(t, sim_nn.detach().numpy(), label='Simulated data', color='red')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.title('Power source prediction')

plt.show()