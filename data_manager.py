import numpy as np
import scipy.io
import pandas as pd
import os
import time
from scipy.io import savemat

class DataManager:
    """
    Manages reading and storing simulation results.
    Tracks UAV state, beamforming, RIS coefficients, user capacity, and jamming power.

    """

    def __init__(self, store_list=None, file_path='./data', store_path='./data/storage'):
        """
        Initializes the data storage system.
        :param store_list: List of variables to store in simulation results.
        """
        self.store_list = ['beamforming_matrix', 'reflecting_coefficient','UAV_state', 'user_capacities_combined', 'G_power', 'user_capacity', 'jamming_power', 'UAV_movement', 'attacker_capacity', 'reward', 'secure_capacity']
        
        self.init_data_file = os.path.join(file_path, 'init_location.xlsx')
        
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.store_path = os.path.join(store_path, self.timestamp)
        os.makedirs(self.store_path, exist_ok=True)

        self.simulation_result_dic = {key: [] for key in store_list}
        self.episode_cnt = 0

    
    def update(self, result):
        for key in self.simulation_result_dic:
            if key in result:
                self.simulation_result_dic[key].append(result[key])
            else:
                print(f"[Warning] Missing key '{key}' in result dictionary.")
    
    # def save_file(self, episode_cnt=0):
    #     """
    #     Saves simulation results to a .mat file after each episode.
    #     """
    #     file_name = os.path.join(self.store_path, f'simulation_result_ep_{episode_cnt}.mat')
    #     scipy.io.savemat(file_name, {f'result_{episode_cnt}': self.simulation_result_dic})
        
    #     # Reset storage for the next episode
    #     self.simulation_result_dic = {key: [] for key in self.store_list}

    def save_file(self, episode_cnt=None):
        if episode_cnt is None:
            episode_cnt = self.episode_cnt
        filename = f"simulation_result_ep_{episode_cnt}.mat"
        save_path = os.path.join(self.store_path, filename)

        mat_struct = {
            f"result_{episode_cnt}": {
                key: [self.simulation_result_dic[key][-1]]  # save only the latest for current episode
                for key in self.simulation_result_dic
            }
        }

        savemat(save_path, mat_struct)
        print(f"[Saved] {save_path}")
        self.episode_cnt += 1

    # def save_meta_data(self, meta_dic):
    #     """
    #     Saves metadata about the system and agent.
    #     """
    #     scipy.io.savemat(os.path.join(self.store_path, 'meta_data.mat'), {'meta_data': meta_dic})

    def save_meta_data(self, meta_dic):
        meta_path = os.path.join(self.store_path, "meta_info.mat")
        savemat(meta_path, meta_dic)
        print(f"[Meta Saved] {meta_path}")


    def read_init_location(self, entity_type='user', index=0):
        """
        Reads initial location data from 'init_location.xlsx'.
        """
        valid_entities = {'user', 'attacker', 'RIS', 'RIS_norm_vec', 'UAV', 'jammer'}
        if entity_type not in valid_entities:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {valid_entities}")

        df = pd.read_excel(self.init_data_file, sheet_name=entity_type)
        return np.array([df.loc[index, 'x'], df.loc[index, 'y'], df.loc[index, 'z']])

    def store_data(self, row_data, value_name):
        """
        Stores simulation data (e.g., beamforming, capacity, jamming power).

        """
        if value_name not in self.simulation_result_dic:
            raise KeyError(f"'{value_name}' is not a valid storage key.")
        
        self.simulation_result_dic[value_name].append(row_data)
