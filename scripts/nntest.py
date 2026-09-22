from hydrogen_rl.paths import DATA_DIR, PLOT_DIR, SOURCE_MODEL_PATH
from hydrogen_rl.sourcefn import function_complex
from hydrogen_rl.sourcemodel import Net
import torch
import matplotlib.pyplot as plt
import numpy as np

# Get from csv both normal data and data with moving average
fd = open(DATA_DIR / "wind.csv", "r")
wind = fd.readlines()
fd.close()
wind = [float(x.strip()) for x in wind[1:]]

fd = open(DATA_DIR / "wind_sv.csv", "r")
wind_sv = fd.readlines()
fd.close()
wind_sv= [float(x.strip()) for x in wind_sv[1:]]

# Number of data points to display
size = 10000

x = [i for i in range(size)]
fun = [function_complex(i) for i in range(size)]

# Plot of original wind speed data and with moving average
plt.figure(figsize=(12, 6))
plt.plot(x, wind[:size], label='Wind speed', color='blue')
plt.plot(x, wind_sv[:size], label='Wind speed smoothed', color='red')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()
plt.savefig(PLOT_DIR / "smoothing.png", dpi=300, bbox_inches="tight")

# Plot of energy availability function compared with qualitative function
plt.figure(figsize=(12, 6))
plt.plot(x, wind_sv[:size], label='Wind speed smoothed', color='red')
plt.plot(x, fun, label='Wind function', color='green')
plt.xlabel('Time')
plt.ylabel('Wind speed')
plt.legend()
plt.savefig(PLOT_DIR / "wind_comparison.png", dpi=300, bbox_inches="tight")

plt.show()

# Load and use the model on real smoothed data and on the qualitative function
model = Net()
model.load_state_dict(torch.load(SOURCE_MODEL_PATH, map_location="cpu"))
model.eval()
power_fn = []
power_sv = []
for i in range(size):
    power_fn.append(model(torch.tensor(np.float32(function_complex(i))).reshape(-1,1)).item())
    power_sv.append(model(torch.tensor(wind_sv[i]).reshape(-1,1)).item())

# Plots of generated power based on wind speed
plt.figure(figsize=(12, 6))
plt.plot(x, power_fn, label='Power given wind speed function', color='blue')
plt.plot(x, power_sv, label='Power given wind speed data (smoothed)', color='red')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.savefig(PLOT_DIR / "power_comparison.png", dpi=300, bbox_inches="tight")

plt.show()