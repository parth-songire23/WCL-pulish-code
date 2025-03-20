import numpy as np
import math
import cmath
from math_tool import *

class mmWave_channel(object):
    """
    Models mmWave channel between transmitter and receiver.
    Supports UAV, RIS, Users, Attackers, and now Jammers.
    """
    def __init__(self, transmitter, receiver, frequency):
        """
        Initializes channel characteristics for a given transmitter-receiver pair.

        :param transmitter: Object from entity.py (UAV, RIS, User, Attacker, Jammer)
        :param receiver: Object from entity.py (UAV, RIS, User, Attacker, Jammer)
        :param frequency: Carrier frequency in Hz (e.g., 28 GHz for mmWave)
        """
        self.channel_name = ''
        self.n = 0  # Path loss exponent
        self.sigma = 0  # Shadow fading standard deviation
        self.transmitter = transmitter
        self.receiver = receiver
        self.channel_type = self.init_type()  # Determines type of channel (e.g., UAV-RIS, UAV-User, etc.)
        
        self.frequency = frequency

        # Initialize and update path loss
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)

        # Initialize and update CSI (Channel State Information)
        self.channel_matrix = self.get_estimated_channel_matrix()

    def init_type(self):
        """
        Identifies the type of communication link based on transmitter and receiver types.
        """
        channel_type = self.transmitter.type + '_' + self.receiver.type

        if channel_type in ['UAV_RIS', 'RIS_UAV']:
            self.n = 2.2  # Path loss exponent for UAV-RIS link
            self.sigma = 3
            self.channel_name = 'H_UR'
        elif channel_type in ['UAV_user', 'user_UAV']:
            self.n = 3.5  # Path loss exponent for UAV-User link
            self.sigma = 3
            self.channel_name = f'h_U_k_{self.transmitter.index}'
        elif channel_type in ['UAV_attacker', 'attacker_UAV']:
            self.n = 3.5  # Path loss exponent for UAV-Attacker link
            self.sigma = 3
            self.channel_name = f'h_U_p_{self.transmitter.index}'
        elif channel_type in ['RIS_user', 'user_RIS']:
            self.n = 2.8  # Path loss exponent for RIS-User link
            self.sigma = 3
            self.channel_name = f'h_R_k_{self.transmitter.index}'
        elif channel_type in ['RIS_attacker', 'attacker_RIS']:
            self.n = 2.8  # Path loss exponent for RIS-Attacker link
            self.sigma = 3
            self.channel_name = f'h_R_p_{self.transmitter.index}'
        elif channel_type in ['UAV_jammer', 'RIS_jammer', 'jammer_UAV', 'jammer_RIS']:
            self.n = 3.0  # Path loss exponent for jamming links
            self.sigma = 3
            self.channel_name = f'Jamming_Link_{self.transmitter.index}'
        return channel_type

    def get_channel_path_loss(self):
        """
        Computes path loss with shadow fading effects.
        """
        distance = np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)

        # Path loss model (log-distance path loss formula)
        PL = -20 * math.log10(4 * math.pi / (3e8 / self.frequency)) - 10 * self.n * math.log10(distance)

        # Shadowing effect (normally distributed)
        shadow_loss = np.random.normal() * self.sigma

        # Convert dB loss to normal scale
        return dB_to_normal(PL - shadow_loss)

    def get_estimated_channel_matrix(self):
        """
        Computes the estimated channel matrix using mmWave modeling.
        """
        N_t = self.transmitter.ant_num  # Number of antennas at transmitter
        N_r = self.receiver.ant_num  # Number of antennas at receiver
        channel_matrix = np.mat(np.ones(shape=(N_r, N_t), dtype=complex), dtype=complex)

        # Compute relative coordinates between transmitter and receiver
        relative_position = self.receiver.coordinate - self.transmitter.coordinate
        r_under_t_car_coor = get_coor_ref(self.transmitter.coor_sys, relative_position)
        t_under_r_car_coor = get_coor_ref([-self.receiver.coor_sys[0], self.receiver.coor_sys[1], -self.receiver.coor_sys[2]], relative_position)

        # Convert to spherical coordinates
        r_t_r, r_t_theta, r_t_fai = cartesian_coordinate_to_spherical_coordinate(r_under_t_car_coor)
        t_r_r, t_r_theta, t_r_fai = cartesian_coordinate_to_spherical_coordinate(t_under_r_car_coor)

        # Compute array responses
        t_array_response = self.generate_array_response(self.transmitter, r_t_theta, r_t_fai)
        r_array_response = self.generate_array_response(self.receiver, t_r_theta, t_r_fai)

        # Compute LOS path loss and phase shift
        PL = self.path_loss_normal
        LOS_fai = 2 * math.pi * self.frequency * np.linalg.norm(relative_position) / 3e8
        channel_matrix = cmath.exp(1j * LOS_fai) * math.sqrt(PL) * (r_array_response * t_array_response.H)

        return channel_matrix

    def generate_array_response(self, transceiver, theta, fai):
        """
        Generates array response vector for different antenna configurations (UPA, ULA, Single).
        """
        ant_type = transceiver.ant_type
        ant_num = transceiver.ant_num

        if ant_type == 'UPA':  # Uniform Planar Array (UPA)
            row_num = int(math.sqrt(ant_num))
            response = np.mat(np.ones(shape=(ant_num, 1)), dtype=complex)
            for i in range(row_num):
                for j in range(row_num):
                    response[j + i * row_num, 0] = cmath.exp(1j * (math.sin(theta) * math.cos(fai) * i * math.pi + math.sin(theta) * math.sin(fai)))
            return response
        elif ant_type == 'ULA':  # Uniform Linear Array (ULA)
            response = np.mat(np.ones(shape=(ant_num, 1)), dtype=complex)
            for i in range(ant_num):
                response[i, 0] = cmath.exp(1j * math.sin(theta) * math.cos(fai) * i * math.pi)
            return response
        elif ant_type == 'single':  # Single Antenna
            return np.mat(np.array([1]))
        else:
            return False

    def update_CSI(self):
        """
        Updates path loss and CSI (channel matrix).
        """
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        self.channel_matrix = self.get_estimated_channel_matrix()
