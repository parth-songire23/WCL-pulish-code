import os
import numpy as np
import matplotlib.pyplot as plt
from env import minimal_IRS_system
from ddpg import Agent

# Fix duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Initialize IRS System with Jammers
IRS_system = minimal_IRS_system(K=1)
K, M, N = IRS_system.K, IRS_system.M, IRS_system.N

# 2. Define RL State & Action Dimensions
RL_state_dims = 2*K + 2*K**2 + 2*N + 2*M*K + 2*N*M + 2*K*N
RL_input_dims = RL_state_dims
RL_action_dims = 2 * (M * K) + N

# 3. Initialize DRL Agent
agent = Agent(
    alpha=0.001, beta=0.001, input_dims=[RL_input_dims], tau=0.001,
    env=IRS_system, batch_size=64, layer1_size=800, layer2_size=600,
    n_actions=RL_action_dims
)

# 4. Training Parameters
episode_num = 1000
steps_per_ep = 200
scores = []

# 5. Training Loop
for i in range(episode_num):
    observersion = IRS_system.reset()
    done = False
    score = 0
    best_bit_per_Hz = 0
    bit_rate_list, power_list = [], []
    
    for step in range(500):  # Prevent infinite loops
        action = agent.choose_action(observersion)
        new_state, reward, done_sys, jamming_effect = IRS_system.step(action)

        # Store performance metrics
        bit_per_Hz = IRS_system.calculate_data_rate()
        total_power = IRS_system.calculate_total_transmit_power()
        bit_rate_list.append(bit_per_Hz)
        power_list.append(total_power)

        # Store best data rate while staying within power limits
        if not done_sys and bit_per_Hz > best_bit_per_Hz:
            best_bit_per_Hz = bit_per_Hz

        # Store experience (including jamming effect)
        agent.remember(observersion, action, reward, new_state, int(done_sys), jamming_effect)
        agent.learn()

        score += reward
        observersion = new_state

        if done_sys:
            break

    # 6. Plot & Save Episode Results
    plt.clf()
    plt.plot(bit_rate_list, color="green", label="Bit Rate (bits/s/Hz)")
    plt.plot(power_list, color="red", label="Total Power (W)")
    plt.legend()
    
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"episode_{i}.png")
    plt.savefig(save_path)

    scores.append(score)

    # 7. Log Training Progress
    print(f"Episode {i}: Score = {score:.2f} | Best Sum Rate = {best_bit_per_Hz:.3f} bits/s/Hz | "
          f"Avg Last 100 = {np.mean(scores[-100:]):.4f}")

    # 8. Save Model Every 50 Episodes
    if i % 50 == 0 and i != 0:
        agent.save_models()
