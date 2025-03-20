import numpy as np
import scipy.io
import pandas as pd
import os
import time

class DataManager(object):
    """
    Handles reading and storing simulation results.
    - Stores UAV state, beamforming, RIS coefficients, user capacity.
    - Now also tracks **jamming power** over episodes.
    """

    def __init__(self, store_list=None, file_path='./data', store_path='./data/storage'):
        """
        Initializes data storage system.
        :param store_list: List of variables to store in simulation results.
        """
        if store_list is None:
            store_list = ['beamforming_matrix', 'reflecting_coefficient', 
                          'UAV_state', 'user_capacity', 'jamming_power']  # Added jamming power

        self.store_list = store_list
        self.init_data_file = os.path.join(file_path, 'init_location.xlsx')

        # Create unique storage directory for each run
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.store_path = os.path.join(store_path, self.timestamp)
        os.makedirs(self.store_path, exist_ok=True)

        self.simulation_result_dic = {}
        self.init_format()

    def save_file(self, episode_cnt=0):
        """
        Saves simulation results to a .mat file after each episode.
        """
        file_name = f'simulation_result_ep_{episode_cnt}.mat'
        scipy.io.savemat(os.path.join(self.store_path, file_name), 
                         {'result_' + str(episode_cnt): self.simulation_result_dic})
        self.simulation_result_dic = {}
        self.init_format()

    def save_meta_data(self, meta_dic):
        """
        Saves metadata about the system and agent.
        """
        scipy.io.savemat(os.path.join(self.store_path, 'meta_data.mat'), 
                         {'meta_data': meta_dic})

    def init_format(self):
        """
        Initializes empty storage dictionary.
        """
        for store_item in self.store_list:
            self.simulation_result_dic[store_item] = []

    def read_init_location(self, entity_type='user', index=0):
        """
        Reads initial location data from 'init_location.xlsx'.
        Now correctly validates entity_type before reading.
        """
        valid_entities = {'user', 'attacker', 'RIS', 'RIS_norm_vec', 'UAV', 'jammer'}
        if entity_type in valid_entities:
            df = pd.read_excel(self.init_data_file, sheet_name=entity_type)
            return np.array([df['x'][index], df['y'][index], df['z'][index]])
        else:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {valid_entities}")

    def store_data(self, row_data, value_name):
        """
        Stores simulation data (e.g., beamforming, capacity, jamming power).
        """
        if value_name in self.simulation_result_dic:
            self.simulation_result_dic[value_name].append(row_data)
        else:
            raise KeyError(f"'{value_name}' is not a valid storage key.")
