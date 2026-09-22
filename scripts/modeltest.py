from hydrogen_rl.env import NetworkEnv, TARGET_POWER
from hydrogen_rl.paths import MODEL_DIR, PLOT_DIR
from hydrogen_rl.sourcefn import function_complex, wind_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error

STEPS = 10000

print("Target power: ", TARGET_POWER)
def testmodel(function, naive = False):

    # Load the environment and model
    env = NetworkEnv(function, naive)
    model = PPO.load(MODEL_DIR / "best_model", env)
    check_env(env, warn=True)

    # Run the model
    obs, _ = env.reset()  

    for i in range(STEPS):
        action, _states = model.predict(obs, deterministic=True)
        next_state, reward, done, truncated, _ = env.step(action)
        obs = next_state

    reward_data, output_data, volume_data, action_data, input_data = env.get_data()

    # Plot input, output, and target data
    target = np.ones(len(output_data)) * TARGET_POWER
    mean = np.mean(output_data)
    output_avg = np.ones(len(output_data))*np.mean(output_data)

    rmse = root_mean_squared_error(output_data, target)
    print(f"\nnaive: {naive} \nsource function: {function.__name__} \nRMSE: {rmse} \nMean: {mean}\n")

    plt.figure(figsize=(12, 6))
    plt.plot(input_data, label="Input")
    plt.plot(output_data, label="Output")
    plt.plot(target, label="Target")
    plt.plot(output_avg, label="Output avg", linestyle='--', color = "gray")
    plt.xlabel("Steps")
    plt.ylabel("Power")
    plt.title("Input and Output")
    plt.legend()
    if naive:
        plt.savefig(PLOT_DIR / ("input_output_naive_" + function.__name__ + ".png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(PLOT_DIR / ("input_output_" + function.__name__ + ".png"), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(volume_data)
    plt.xlabel('Steps')
    plt.ylabel('Volume')
    plt.title("Energy storage volume")
    if naive:
        plt.savefig(PLOT_DIR / ("volume_naive_" + function.__name__ + ".png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(PLOT_DIR / ("volume_output_" + function.__name__ + ".png"), dpi=300, bbox_inches='tight')

    if naive == False:
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(action_data)
        plt.xlabel('Steps')
        plt.ylabel('Action')
        plt.title("Actions")
        plt.legend(["Pow to H2", "H2 to Pow"])
        plt.savefig(PLOT_DIR / ("actions_" + function.__name__ + ".png"), dpi=300, bbox_inches='tight')

    plt.show()

    env.close()

if __name__ == "__main__":
    testmodel(function_complex)
    testmodel(function_complex, True)
    testmodel(wind_data)
    testmodel(wind_data, True)