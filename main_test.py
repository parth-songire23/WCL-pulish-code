import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from env import MiniSystem
from ddpy import Agent
import numpy as np
import time

# 1. Initialize System with Jammers
system = MiniSystem(
    user_num=2,
    attacker_num=1,  # Ensure attacker is included
    RIS_ant_num=4,
    UAV_ant_num=4,
    if_dir_link=1,
    if_with_RIS=True,
    if_move_users=True,
    if_movements=True,
    reverse_x_y=(False, False),
    if_UAV_pos_state=True
)

# 2. Training Parameters
episode_num = 100
step_num = 100
episode_cnt = 0

# 3. Define RL Agents for UAV and RIS
agent_params = {
    "alpha": 0.0001,
    "beta": 0.001,
    "tau": 0.001,
    "batch_size": 64,
    "memory_max_size": episode_num * step_num,
}

agent_1 = Agent(
    **agent_params,
    input_dims=[system.get_system_state_dim()],
    env=system,
    n_actions=system.get_system_action_dim() - 2,
    layer1_size=800, layer2_size=600, layer3_size=512, layer4_size=256,
    agent_name="G_and_Phi"
)

agent_2 = Agent(
    **agent_params,
    input_dims=[3],
    env=system,
    n_actions=2,
    layer1_size=400, layer2_size=300, layer3_size=256, layer4_size=128,
    agent_name="UAV"
)

# 4. Save System Metadata
meta_dic = {
    "folder_name": system.data_manager.timestamp,
    "user_num": system.user_num,
    "if_dir_link": system.if_dir_link,
    "if_with_RIS": system.if_with_RIS,
    "RIS_ant_num": system.RIS.ant_num,
    "UAV_ant_num": system.UAV.ant_num,
    "episode_num": episode_num,
    "step_num": step_num,
}
system.data_manager.save_meta_data(meta_dic)

# 5. Training Loop
while episode_cnt < episode_num:
    system.reset()  # Reset environment at start of episode
    step_cnt = 0
    score_per_ep = 0

    # Get Initial States
    observersion_1 = system.observe()
    observersion_2 = list(system.UAV.coordinate)

    while step_cnt < step_num:
        step_cnt += 1

        # Select Actions
        action_1 = agent_1.choose_action(observersion_1, greedy=0.1)
        action_2 = agent_2.choose_action(observersion_2, greedy=0.5)

        # Take Step in Environment
        new_state_1, reward, done, jamming_effect = system.step(
            action_0=action_2[0],
            action_1=action_2[1],
            G=action_1[:2 * system.UAV.ant_num * system.user_num],
            Phi=action_1[2 * system.UAV.ant_num * system.user_num:],
            set_pos_x=action_2[0],
            set_pos_y=action_2[1]
        )
        new_state_2 = list(system.UAV.coordinate)

        score_per_ep += reward

        # Store Transitions in Replay Buffer (Including Jamming Effect)
        agent_1.remember(observersion_1, action_1, reward, new_state_1, done, jamming_effect)
        agent_2.remember(observersion_2, action_2, reward, new_state_2, done, jamming_effect)

        # Train Agents
        agent_1.learn()
        agent_2.learn()

        # Update Observations
        observersion_1, observersion_2 = new_state_1, new_state_2

        # Check if Episode is Done
        if done:
            break

    # Save Episode Results
    system.data_manager.save_file(episode_cnt=episode_cnt)
    print(f"Episode {episode_cnt}: Score = {score_per_ep:.2f}")

    episode_cnt += 1

    # Save Model Every 10 Episodes
    if episode_cnt % 10 == 0:
        agent_1.save_models()
        agent_2.save_models()
