import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .paths import DATA_DIR, PLOT_DIR, SOURCE_MODEL_PATH

N_EPOCHS = 2000

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 64)  # 1 neuron in input layer, 64 neurons in 1st hidden layer 1
        self.hidden2 = torch.nn.Linear(64, 128)  # 128 neurons in 2nd hidden layer
        self.output = torch.nn.Linear(128, 1)  # 1 neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.output(x))  # Necessary to clip any negative values
        return x


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    data = pd.read_csv(DATA_DIR / "T1.csv")
    data.head()

    cols = ["Date/Time", "Theoretical_Power_Curve (KWh)", "Wind Direction (°)"]
    data = data.drop(cols, axis=1)
    data.head()

    print(data.shape)

    x_train = data.iloc[1:, 1].values
    y_train = data.iloc[1:, 0].values
    x_train = np.clip(x_train, 0, None).reshape(-1,1)  # Transform negative readings to 0
    y_train = np.clip(y_train, 0, None).reshape(-1,1)

    net = Net()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    print(x_train.shape, y_train.shape)

    loss_values = []

    for epoch in range(N_EPOCHS):
        running_loss = 0.0
        optimizer.zero_grad()
        outputs = net(torch.tensor(x_train).float())
        loss = criterion(outputs, torch.tensor(y_train).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % 50 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))
            loss_values.append(loss.detach().numpy())

    x_plot = [i for i in range(1000)]
    actual_y = [y_train[i] for i in range(len(x_plot))]
    x_subset = torch.tensor(x_train[: len(x_plot)]).float()
    predicted_y = net(x_subset).squeeze()


    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, actual_y, "g", label="Actual Function")
    plt.plot(x_plot, predicted_y.detach().numpy(), "b", label="Predicted Function")
    plt.legend()
    plt.savefig(PLOT_DIR / "prediction.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, N_EPOCHS, N_EPOCHS//50), loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss function")
    plt.savefig(PLOT_DIR / "loss.png", dpi=300, bbox_inches="tight")

    # Overall function
    x_min = x_train.min()
    x_max = x_train.max()
    x_plot = torch.linspace(x_min, x_max, 1000).reshape(-1, 1)
    predicted_y = net(x_plot).squeeze()

    plt.figure(figsize=(12, 6))
    plt.scatter(x_train, y_train, color = "green", label="Actual Function")
    plt.scatter(x_plot, predicted_y.detach().numpy(), color="blue", label="Predicted Function")
    plt.legend()
    plt.xlabel('Wind speed')
    plt.ylabel('Power')
    plt.savefig(PLOT_DIR / "overall.png", dpi=300, bbox_inches="tight")

    plt.show()
    torch.save(net.state_dict(), SOURCE_MODEL_PATH)


if __name__ == "__main__":
    main()
