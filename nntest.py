import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sourcemodel import Net

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
plt.legend()

plt.show()