# HydrogenRL

Reinforcement learning for managing hydrogen storage connected to a wind energy source. The PPO agent controls the conversion between electrical energy and hydrogen to keep the output power close to the target.

The project follows the pipeline described in [Relazione.pdf](Relazione.pdf):

1. a neural network approximates the generated power as a function of wind speed;
2. `NetworkEnv` simulates the source, electrolyzer, tank, and combustor;
3. PPO learns a policy on continuous facility actions;
4. test scripts produce plots and comparisons with the naive heuristic.

## Requirements

- Python 3.10 or higher
- Sufficient CPU for PPO training; CUDA is optional
- Approximately 1 GB free for environment and dependencies

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The installation uses PyTorch compatible with your platform. For a specific CUDA build, first install the PyTorch version indicated by the official PyTorch selector, then run `python -m pip install -e .`.

Alternatively, you can directly use `requirements.txt` inside a virtual environment.

## Reproducible Training

Complete training from scratch requires first creating the source model:

```bash
# 1. Train the wind -> power network and save artifacts/models/sourcefn_model.pth
python -m hydrogen_rl.sourcemodel

# 2. Train PPO for 1,000,000 timesteps and save the best model in artifacts/models/
python -m scripts.agentPPO
```

The second command does not load previous checkpoints: it initializes PPO from scratch. Training plots are written to `artifacts/plots/`.

To evaluate the saved model:

```bash
python -m scripts.modeltest
```

To run auxiliary studies:

```bash
python -m scripts.nntest
python -m scripts.reward_study
```

Scripts require `artifacts/models/sourcefn_model.pth` to exist; `scripts.modeltest` also requires `artifacts/models/best_model.zip`. Artifacts already present in the repository allow you to skip training, but are not necessary for a new run from scratch.

## Project Structure

```text
HydrogenRL/
├── data/                         # CSV dataset and wind series
├── src/hydrogen_rl/              # importable Python package
│   ├── env.py                    # Gymnasium environment and physical components
│   ├── sourcefn.py               # wind availability functions
│   ├── sourcemodel.py            # neural network wind -> power
│   └── paths.py                  # shared project paths
├── scripts/                      # entry points and analysis
│   ├── agentPPO.py               # PPO training from scratch
│   ├── modeltest.py              # PPO evaluation and naive baseline
│   ├── nntest.py                 # source network plots
│   └── reward_study.py           # reward heatmap
├── artifacts/
│   ├── models/                   # PyTorch and PPO checkpoints
│   └── plots/                    # generated plots
├── Relazione.pdf                 # original technical documentation
├── pyproject.toml                # package metadata and dependencies
└── requirements.txt              # pip-installable dependencies
```

Paths are resolved relative to the project root, so commands work even if run from a different directory, provided the package is installed with `pip install -e .`.

## Model and Environment

The source network is an MLP `1 -> 64 -> 128 -> 1`, with ReLU, Adam, MSE, and 2000 epochs. The data uses active power as target and wind speed as input; negative values are clipped to zero.

`NetworkEnv` exposes:

- **observation**: `[normalized_source_power, normalized_tank_volume]`;
- **action**: `[power_to_store_fraction, hydrogen_to_release_fraction]`, initially in `[-1, 1]` and converted to `[0, 1]` by the environment;
- **reward**: combination of target error, stability between consecutive steps, and incentive for storage during surplus;
- **episode end**: after 10000 steps, without natural terminal state.

The main physical parameters are defined in `src/hydrogen_rl/env.py`.

## Notes

The model is a simplified mathematical simulation: it does not replace a complete physical model of electrolyzer, tank, or fuel cell. Results should therefore be interpreted as a control study and not as sizing of a real facility.
