import os
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden1 = torch.nn.Linear(1, 64) # 1 neuron in input layer, 64 neurons in 1st hidden layer 1
    self.hidden2 = torch.nn.Linear(64, 128) # 128 neurons in 2nd hidden layer
    self.output = torch.nn.Linear(128, 1) # 1 neuron in output layer

  def forward(self, x):
    x = torch.relu(self.hidden1(x))
    x = torch.relu(self.hidden2(x))
    x = self.output(x)
    return x

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Using {} device'.format(device))

  data = pd.read_csv("csv/T1.csv")
  data.head()

  cols = ["Date/Time", "Theoretical_Power_Curve (KWh)", "Wind Direction (°)"]
  data = data.drop(cols, axis=1)
  data.head()

  print(data.shape)

  x_train = data.iloc[1:, 1].values
  y_train = data.iloc[1:, 0].values
  x_train = x_train.reshape(-1, 1)
  y_train = y_train.reshape(-1, 1)

  net = Net()
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

  print(x_train.shape, y_train.shape)

  for epoch in range(1500):
    running_loss = 0.0
    optimizer.zero_grad()
    outputs = net(torch.tensor(x_train).float())
    loss = criterion(outputs, torch.tensor(y_train).float())
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if epoch % 100 == 0:
        print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

  x_plot = [i for i in range(1000)]
  actual_y = [y_train[i] for i in range(len(x_plot))]
  x_subset = torch.tensor(x_train[:len(x_plot)]).float()
  predicted_y = net(x_subset).squeeze()

  plt.figure(figsize=(12, 6))
  plt.plot(x_plot, actual_y, 'g', label='Actual Function')
  plt.plot(x_plot, predicted_y.detach().numpy(), 'b', label='Predicted Function')
  plt.legend()
  plt.savefig("sourcefn_plots/prediction.png", dpi=300, bbox_inches='tight')

  #overall function
  x_min = x_train.min()
  x_max = x_train.max()
  x_plot = torch.linspace(x_min, x_max, 1000).reshape(-1, 1)
  predicted_y = net(x_plot).squeeze()

  plt.figure(figsize=(12, 6))
  plt.scatter(x_train, y_train, color='green', marker='o', label='Actual Function')
  plt.plot(x_plot, predicted_y.detach().numpy(), 'b', label='Predicted Function')
  plt.legend()
  plt.savefig("sourcefn_plots/approximation.png", dpi=300, bbox_inches='tight')

  plt.show()

  torch.save(net.state_dict(), "sourcefn_model.pth")


if __name__ == "__main__":
    main()