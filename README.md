# IRS-Assisted UAV Communication with Anti-Jamming: DRL Approach

## Overview
This repository implements Deep Reinforcement Learning (DRL) for optimizing IRS-assisted UAV communication in the presence of jamming attacks. It extends previous research on secure UAV communication by introducing a jamming entity to study its effects and optimize UAV behavior accordingly.

## Repository Structure
Each file in this repository serves a specific purpose in the simulation and training of the DRL model.

| **File** | **Description** |
|----------|----------------|
| `env.py` | Defines the simulation environment, including UAV, RIS, Users, Attackers, and Jammers. This is the core system model. |
| `entity.py` | Contains the definitions of entities such as UAV, RIS, Users, Attackers, and the newly added Jammer. |
| `channel.py` | Implements the mmWave channel model, including path loss calculations and interference from the Jammer. |
| `ddpg.py` | Contains the Deep Deterministic Policy Gradient (DDPG) algorithm, used for DRL-based optimization of UAV trajectory and beamforming. |
| `data_manager.py` | Handles data storage, saving results in `.mat` format, and initializing system parameters from `init_location.xlsx`. |
| `render.py` | Provides a 3D visualization of the simulation, including UAV, RIS, and jamming effects. |
| `test_channel.py` | A simple script to test the mmWave channel initialization and jamming effects. |
| `main_test.py` | The main training script for the DRL model, simulating UAV communication with RIS under jamming. |
| `main_ref.py` | Similar to `main_test.py`, but used for testing different DRL parameters and comparing results. |
| `main_RIS.py` | Focuses on RIS configuration testing and analyzing the impact of RIS elements on communication quality. |
| `load_and_plot.py` | Loads saved data and plots results for SSR, SEE, UAV trajectory, and jamming interference. |
| `batch_train.sh` | A shell script to train DRL models in batch mode. |
| `batch_eval.sh` | A shell script to evaluate trained models based on stored results. |

## Order of Execution
Follow these steps to understand and run this repository.

### Step 1: Environment Setup
1. Clone the Repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
2. Create and Activate Virtual Environment
   ```bash
   conda create --name uav_env python=3.10
   conda activate uav_env
3. Install Dependencies
   ```bash
   pip install -r requirements.txt

   
### Step 2: Understanding the Code
1. Test UAV-RIS Communication
   Run test_channel.py to verify that the UAV and RIS communicate correctly:

   ```bash
   python3 test_channel.py
2. Visualize the UAV-RIS-Jammer System
   Run render.py to generate a 3D visualization of the system:
   ```bash
   python3 render.py
3. Analyze Existing Results
   Run load_and_plot.py to plot stored simulation results:
   ```bash
   python3 load_and_plot.py --path ./data/storage/


### Step 3: Training the DRL Model
1. Train the DRL Model
   Train the UAV trajectory and beamforming optimization using:
   ```bash
   python3 main_test.py
2. Train with Reference Parameters
   To compare different DRL parameter settings, run:
   ```bash
   python3 main_ref.py
3. Run RIS-Specific Testing
   To analyze the effect of RIS on UAV communication:
   ```bash
   python3 main_RIS.py


### Step 4: Running Simulations on Trained Models
1. Run Simulations Using Trained Models
   ```bash
   python3 run_simulation.py --path data/storage/scratch/
2. Evaluate DRL Model Performance in Batch Mode
   ```bash
   bash batch_eval.sh

