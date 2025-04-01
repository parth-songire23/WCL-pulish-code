import numpy as np
import math
import cmath
from math_tool import *

class mmWave_Channel:
    """
    Models mmWave channel between transmitter and receiver.
    Supports UAV, RIS, Users, Attackers, and Jammers.
    """
    def __init__(self, transmitter, receiver, frequency):
        self.transmitter = transmitter
        self.receiver = receiver
        self.frequency = frequency
        
        self.channel_type, self.channel_name, self.n, self.sigma = self.init_type()
        
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        self.channel_matrix = self.get_estimated_channel_matrix()

    def init_type(self):
        """Identifies the communication link type based on transmitter and receiver."""
        channel_type = f"{self.transmitter.type}_{self.receiver.type}"
        type_mapping = {
            ('UAV', 'RIS'): (2.2, 'H_UR'), ('RIS', 'UAV'): (2.2, 'H_UR'),
            ('UAV', 'user'): (3.5, 'h_U_k'), ('user', 'UAV'): (3.5, 'h_U_k'),
            ('UAV', 'attacker'): (3.5, 'h_U_p'), ('attacker', 'UAV'): (3.5, 'h_U_p'),
            ('RIS', 'user'): (2.8, 'h_R_k'), ('user', 'RIS'): (2.8, 'h_R_k'),
            ('RIS', 'attacker'): (2.8, 'h_R_p'), ('attacker', 'RIS'): (2.8, 'h_R_p'),
            ('UAV', 'jammer'): (3.0, 'Jamming_Link'), ('RIS', 'jammer'): (3.0, 'Jamming_Link'),
            ('jammer', 'UAV'): (3.0, 'Jamming_Link'), ('jammer', 'RIS'): (3.0, 'Jamming_Link')
        }
        n, base_name = type_mapping.get((self.transmitter.type, self.receiver.type), (3.0, 'Unknown'))
        channel_name = f"{base_name}_{self.transmitter.index}"
        return channel_type, channel_name, n, 3  # Default shadow fading std. dev.

    def get_channel_path_loss(self):
        """Computes path loss with shadow fading effects."""
        distance = np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)
        path_loss_dB = -20 * math.log10(4 * math.pi * self.frequency / 3e8) - 10 * self.n * math.log10(distance)
        shadowing = np.random.normal() * self.sigma
        return dB_to_normal(path_loss_dB - shadowing)

    def get_estimated_channel_matrix(self):
        """Computes estimated channel matrix for mmWave propagation."""
        N_t, N_r = self.transmitter.ant_num, self.receiver.ant_num
        channel_matrix = np.mat(np.ones((N_r, N_t), dtype=complex))
        
        relative_position = self.receiver.coordinate - self.transmitter.coordinate
        r_under_t_car_coor = get_coor_ref(self.transmitter.coor_sys, relative_position)
        t_under_r_car_coor = get_coor_ref([-self.receiver.coor_sys[0], self.receiver.coor_sys[1], -self.receiver.coor_sys[2]], relative_position)
        
        r_t_r, r_t_theta, r_t_fai = cartesian_coordinate_to_spherical_coordinate(r_under_t_car_coor)
        t_r_r, t_r_theta, t_r_fai = cartesian_coordinate_to_spherical_coordinate(t_under_r_car_coor)
        
        t_array_response = self.generate_array_response(self.transmitter, r_t_theta, r_t_fai)
        r_array_response = self.generate_array_response(self.receiver, t_r_theta, t_r_fai)
        
        LOS_phase = 2 * math.pi * self.frequency * np.linalg.norm(relative_position) / 3e8
        return cmath.exp(1j * LOS_phase) * math.sqrt(self.path_loss_normal) * (r_array_response * t_array_response.H)

    def generate_array_response(self, transceiver, theta, fai):
        """Generates array response vector based on antenna configuration."""
        ant_type, ant_num = transceiver.ant_type, transceiver.ant_num
        response = np.mat(np.ones((ant_num, 1), dtype=complex))
        
        if ant_type == 'UPA':
            row_num = int(math.sqrt(ant_num))
            for i in range(row_num):
                for j in range(row_num):
                    response[j + i * row_num, 0] = cmath.exp(1j * (math.sin(theta) * math.cos(fai) * i * math.pi + math.sin(theta) * math.sin(fai)))
        elif ant_type == 'ULA':
            for i in range(ant_num):
                response[i, 0] = cmath.exp(1j * math.sin(theta) * math.cos(fai) * i * math.pi)
        return response if ant_type in ['UPA', 'ULA'] else np.mat(np.array([1]))

    def update_CSI(self):
        """Updates path loss and channel state information (CSI)."""
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        self.channel_matrix = self.get_estimated_channel_matrix()
