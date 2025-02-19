import gymnasium as gym
import numpy as np
import traci  # Assicurati che libsumo sia installato correttamente
import random
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from icecream import ic
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

import libsumo as traci
from gymnasium import spaces
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console


class RichProgressBar:
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[progress.elapsed] Time elapsed: {task.elapsed}"),
            TextColumn("[progress.remaining] Time remaining: {task.remaining}"),
            console=self.console,
        )
        self.task = None

    def start(self, total_timesteps):
        self.task = self.progress.add_task("[cyan]Training...", total=total_timesteps)
        self.progress.start()

    def update(self, completed_timesteps):
        if self.task:
            self.progress.update(self.task, completed=completed_timesteps)

    def finish(self):
        if self.task:
            self.progress.stop()


class SumoBatteryEnv(gym.Env):
    def __init__(self, gui=False):
        super(SumoBatteryEnv, self).__init__()

        self.gui = gui
        self.step_count = 0
        self.vehicle_entered = False  # Track if the vehicle has entered the simulation
        self.max_speed = 30  # Ad esempio, 30 m/s
        self.max_acceleration = 5
        # Define action space: continuous actions for speed and acceleration
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # Lower bound for speed and acceleration
            high=np.array([1, 1]),  # Upper bound for speed and acceleration
            dtype=np.float32,
        )

        # Define observation space: battery, energy consumed, speed, gap, acceleration
        low = np.array([0, 0, -np.inf, 0, -np.inf], dtype=np.float32)
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Command to run SUMO
        self.sumoCmd = [
            "sumo-gui" if self.gui else "sumo",
            "-c",
            "iter/19/iteration_019.sumocfg",
            "--start",
            "--quit-on-end",
        ]

        self.rew_ep = []  # Store rewards
        self.speed_data = []  # Store speed data
        self.acceleration_data = []  # Store acceleration data
        self.energy_data = []  # Store energy consumed data
        self.position_data = []
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Seed for reproducibility
        traci.start(self.sumoCmd)
        edge_ids = traci.edge.getIDList()
        edge_ids = [el for el in edge_ids if not el.startswith(":")]
        route = random.sample(edge_ids, 2)
        traci.route.add("trip", route)
        traci.vehicle.add(vehID="001_myego", routeID="trip", typeID="ev", depart="now")
        self.step_count = 0
        self.vehicle_entered = False  # Reset the flag
        self.speed_data.clear()  # Clear stored speed data
        self.acceleration_data.clear()  # Clear stored acceleration data
        self.energy_data.clear()
        self.position_data.clear()# Clear stored energy data
        return self._get_state(), {}  # Return state and empty info

    def step(self, action):
        my_ego = "001_myego"

        # Check if the vehicle is in the simulation
        if my_ego in traci.vehicle.getIDList():
            self.vehicle_entered = True

        if self.vehicle_entered and my_ego not in traci.vehicle.getIDList():
            ic("vehicle exited simulation, resetting environment")
            return self.reset()

        self._apply_action(action)
        traci.simulationStep()
        self.step_count += 1

        if my_ego in traci.vehicle.getIDList():
            state = self._get_state()
            reward = self._get_reward()
            done = self._is_done()
            truncated = self._is_truncated()
            info = {}
            #ic(traci.simulation.getTime())
            self._collect()

            # Update SUMO GUI
            if self.gui:
                traci.gui.setZoom("View #0", 200)
                traci.gui.trackVehicle("View #0", my_ego)
        else:
            ic("vehicle not in simulation")
            state = np.array([0, 0, 0, 0, 0], dtype=np.float32)
            reward = 0
            done = False
            truncated = True
            info = {}

        return state, reward, done, truncated, info

    def _get_state(self):
        my_ego = "001_myego"
        battery = float(
            traci.vehicle.getParameter(my_ego, "device.battery.actualBatteryCapacity")
        )
        energy_consumed = float(
            traci.vehicle.getParameter(my_ego, "device.battery.totalEnergyConsumed")
        )
        speed = traci.vehicle.getSpeed(my_ego)
        gap = traci.vehicle.getMinGap(my_ego)
        acceleration = traci.vehicle.getAcceleration(my_ego)

        state = np.array(
            [battery, energy_consumed, speed, gap, acceleration], dtype=np.float32
        )
        # ic(state)
        return state

    def _apply_action(self, action):
        my_ego = "001_myego"
        action = np.clip(
            action, -1, 1
        )  # Assicurati che l'azione sia all'interno del range [-1, 1]

        current_speed = traci.vehicle.getSpeed(my_ego)
        delta_speed = action[0] * self.max_speed  # Scale the action for speed change
        acceleration = max(
            0.1, abs(action[1] * self.max_acceleration)
        )  # Scale acceleration

        if acceleration != 0:
            duration = abs(delta_speed) / acceleration  # Compute duration
        else:
            duration = 1

        new_speed = max(0, current_speed + delta_speed)
        #ic(new_speed)
        traci.vehicle.slowDown(my_ego, new_speed, duration)

    def _get_reward(self):
        my_ego = "001_myego"
        state = self._get_state()
        current_speed = state[2]
        acceleration = state[4]
        energy_consumed = float(
            traci.vehicle.getParameter(my_ego, "device.battery.totalEnergyConsumed")
        )
        speed_penalty = 0 if current_speed > 5 else -20  # Imposta una penalità per velocità sotto 5 m/s
        acc_pen = -np.abs(acceleration)  # Imposta una penalità per velocità sotto 5 m/s


        # Reward function
        reward = (
            np.log(1 + (np.float32(current_speed) / 13.89))
            - np.exp(np.float32(acceleration)/5) ** 2
            - np.log(1 + np.float32(energy_consumed))
            + acc_pen
            + speed_penalty
        )
        #ic(reward)
        return reward

    def _is_done(self):
        my_ego = "001_myego"
        return my_ego in traci.simulation.getArrivedIDList()

    def _is_truncated(self):
        return self.step_count >= 271

    def close(self):
        traci.close()

    def _collect(self):
        state = self._get_state()
        self.rew_ep.append(self._get_reward())
        self.speed_data.append(state[2])
        self.acceleration_data.append(state[4])
        self.energy_data.append(state[1])
        self.position_data.append(traci.vehicle.getPosition("001_myego"))

    def get_res(self):
        return self.rew_ep, self.speed_data, self.acceleration_data, self.energy_data, self.position_data
    def get_route(self):
        # Get the current route the vehicle is taking
        my_ego = "001_myego"
        if my_ego in traci.vehicle.getIDList():
            route = traci.vehicle.getRoute(my_ego)
            return route
        return []

def main():
    ic.enable()
    env = SumoBatteryEnv(gui=False)

    # Check environment
    check_env(env, warn=True)

    # Use PPO for training
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2_00_000, progress_bar=RichProgressBar())

    # Save the model
    model.save("ppo_sumo_battery")

    # Load the model
    # model = PPO.load('ppo_sumo_battery')

    # Plot rewards, speed, acceleration, and energy consumed
    rewards, speeds, accelerations, energies , positions = env.get_res()

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(3, 1, 2)
    plt.plot(speeds, label="Speed")
    plt.plot(accelerations, label="Acceleration")
    plt.title("Speed and Acceleration")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(energies)
    plt.title("Energy Consumed")
    plt.tight_layout()
    plt.show()
   

    x_values = [x for x, _ in positions]
    y_values = [y for _, y in positions]
    plt.figure(figsize=(14, 10))

    plt.plot(x_values, y_values)
    # Close environment
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
