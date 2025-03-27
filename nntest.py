from sourcefn import function_complex
from sourcemodel import Net
import torch
import matplotlib.pyplot as plt
import numpy as np

# Ottengo dal csv sia i dati normali che quelli con media mobile
fd = open("csv/wind.csv", "r")
wind = fd.readlines()
fd.close()
wind = [float(x.strip()) for x in wind[1:]]

fd = open("csv/wind_sv.csv", "r")
wind_sv = fd.readlines()
fd.close()
wind_sv= [float(x.strip()) for x in wind_sv[1:]]

# Numero di dati da visualizzare
size = 10000

x = [i for i in range(size)]
fun = [function_complex(i) for i in range(size)]

# Grafico dei dati di velocità del vento originali e con media mobile
plt.figure(figsize=(12, 6))
plt.plot(x, wind[:size], label='Wind speed', color='blue')
plt.plot(x, wind_sv[:size], label='Wind speed smoothed', color='red')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()
plt.savefig("sourcefn_plots/smoothing.png", dpi=300, bbox_inches='tight')

# Grafico della funzione di disponibilità di energia confrontata con funzione qualitativa
plt.figure(figsize=(12, 6))
plt.plot(x, wind_sv[:size], label='Wind speed smoothed', color='red')
plt.plot(x, fun, label='Wind function', color='green')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()
plt.savefig("sourcefn_plots/wind_comparison.png", dpi=300, bbox_inches='tight')

plt.show()

# Caricamento e utilizzo del modello sui dati reali smoothed e sulla funzione qualitativa
model = Net()
model.load_state_dict(torch.load("sourcefn_model.pth"))
model.eval()
power_fn = []
power_sv = []
for i in range(size):
    power_fn.append(model(torch.tensor(np.float32(function_complex(i))).reshape(-1,1)).item())
    power_sv.append(model(torch.tensor(wind_sv[i]).reshape(-1,1)).item())

# Grafici della potenza generata in base alla velocità del vento
plt.figure(figsize=(12, 6))
plt.plot(x, power_fn, label='Power given wind speed function', color='blue')
plt.plot(x, power_sv, label='Power given wind speed data (smoothed)', color='red')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.savefig("sourcefn_plots/power_comparison.png", dpi=300, bbox_inches='tight')

plt.show()