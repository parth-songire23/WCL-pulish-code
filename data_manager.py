import numpy as np
import scipy.io
import pandas as pd
import os
import time

class DataManager(object):
    """
    Class to read and store simulation results.
    Before use, please create a directory under the current file path: './data'
    It must include the file 'init_location.xlsx' which contains positions of all entities.
    """
    def __init__(self, store_list=None, file_path='./data', store_path='./data/storage'):
        # 1. Initialize storage
        if store_list is None:
            store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity']
        self.store_list = store_list

        # 2. Read location file
        self.init_data_file = os.path.join(file_path, 'init_location.xlsx')

        # 3. Setup unique storage folder with timestamp
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.store_path = os.path.join(store_path, timestamp)
        os.makedirs(self.store_path, exist_ok=True)

        # 4. Initialize data dictionary
        self.simulation_result_dic = {}
        self.init_format()

    def save_file(self, episode_cnt=0):
        """
        Saves the simulation results to a .mat file at the end of an episode.
        """
        filename = os.path.join(self.store_path, f'simulation_result_ep_{episode_cnt}.mat')
        scipy.io.savemat(filename, {f'result_{episode_cnt}': self.simulation_result_dic})
        self.simulation_result_dic = {}
        self.init_format()

    def save_meta_data(self, meta_dic):
        """
        Saves metadata such as system parameters and configuration.
        """
        filepath = os.path.join(self.store_path, 'meta_data.mat')
        scipy.io.savemat(filepath, {'meta_data': meta_dic})

    def init_format(self):
        """
        Initializes the simulation result dictionary with empty lists for each store item.
        """
        for store_item in self.store_list:
            self.simulation_result_dic[store_item] = []

    def read_init_location(self, entity_type='user', index=0):
        """
        Reads initial coordinates (x, y, z) of an entity from Excel.
        """
        valid_entities = ['user', 'attacker', 'RIS', 'RIS_norm_vec', 'UAV']
        if entity_type not in valid_entities:
            raise ValueError(f"'{entity_type}' is not a valid entity type. Must be one of {valid_entities}")
        
        df = pd.read_excel(self.init_data_file, sheet_name=entity_type)
        return np.array([df.at[index, 'x'], df.at[index, 'y'], df.at[index, 'z']])

    def store_data(self, row_data, value_name):
        """
        Appends data to the simulation result dictionary under the specified key.
        Initializes the key if it doesn't exist.
        """
        if value_name not in self.simulation_result_dic:
            self.simulation_result_dic[value_name] = []
        self.simulation_result_dic[value_name].append(row_data)
