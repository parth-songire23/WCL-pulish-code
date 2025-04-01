# import numpy as np
# import math
# import cmath
# from math_tool import *

# class mmWave_channel:
#     """
#     Models mmWave channel between transmitter and receiver.
#     Supports UAV, RIS, Users, Attackers, and Jammers.
#     """
#     def __init__(self, transmitter, receiver, frequency):
#         self.transmitter = transmitter
#         self.receiver = receiver
#         self.frequency = frequency
        
#         self.channel_type, self.channel_name, self.n, self.sigma = self.init_type()
        
#         self.path_loss_normal = self.get_channel_path_loss()
#         self.path_loss_dB = normal_to_dB(self.path_loss_normal)
#         self.channel_matrix = self.get_estimated_channel_matrix()

#     def init_type(self):
#         """Identifies the communication link type based on transmitter and receiver."""
#         channel_type = f"{self.transmitter.type}_{self.receiver.type}"
#         type_mapping = {
#             ('UAV', 'RIS'): (2.2, 'H_UR'), ('RIS', 'UAV'): (2.2, 'H_UR'),
#             ('UAV', 'user'): (3.5, 'h_U_k'), ('user', 'UAV'): (3.5, 'h_U_k'),
#             ('UAV', 'attacker'): (3.5, 'h_U_p'), ('attacker', 'UAV'): (3.5, 'h_U_p'),
#             ('RIS', 'user'): (2.8, 'h_R_k'), ('user', 'RIS'): (2.8, 'h_R_k'),
#             ('RIS', 'attacker'): (2.8, 'h_R_p'), ('attacker', 'RIS'): (2.8, 'h_R_p'),
#             ('UAV', 'jammer'): (3.0, 'Jamming_Link'), ('RIS', 'jammer'): (3.0, 'Jamming_Link'),
#             ('jammer', 'UAV'): (3.0, 'Jamming_Link'), ('jammer', 'RIS'): (3.0, 'Jamming_Link')
#         }
#         n, base_name = type_mapping.get((self.transmitter.type, self.receiver.type), (3.0, 'Unknown'))
#         channel_name = f"{base_name}_{self.transmitter.index}"
#         return channel_type, channel_name, n, 3  # Default shadow fading std. dev.

#     def get_channel_path_loss(self):
#         """Computes path loss with shadow fading effects."""
#         distance = np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)
#         path_loss_dB = -20 * math.log10(4 * math.pi * self.frequency / 3e8) - 10 * self.n * math.log10(distance)
#         shadowing = np.random.normal() * self.sigma
#         return normal_to_dB(path_loss_dB - shadowing)

#     def get_estimated_channel_matrix(self):
#         """Computes estimated channel matrix for mmWave propagation."""
#         N_t, N_r = self.transmitter.ant_num, self.receiver.ant_num
#         channel_matrix = np.mat(np.ones((N_r, N_t), dtype=complex))
        
#         relative_position = self.receiver.coordinate - self.transmitter.coordinate
#         r_under_t_car_coor = get_coor_ref(self.transmitter.coor_sys, relative_position)
#         t_under_r_car_coor = get_coor_ref([-self.receiver.coor_sys[0], self.receiver.coor_sys[1], -self.receiver.coor_sys[2]], relative_position)
        
#         r_t_r, r_t_theta, r_t_fai = cartesian_coordinate_to_spherical_coordinate(r_under_t_car_coor)
#         t_r_r, t_r_theta, t_r_fai = cartesian_coordinate_to_spherical_coordinate(t_under_r_car_coor)
        
#         t_array_response = self.generate_array_response(self.transmitter, r_t_theta, r_t_fai)
#         r_array_response = self.generate_array_response(self.receiver, t_r_theta, t_r_fai)
        
#         LOS_phase = 2 * math.pi * self.frequency * np.linalg.norm(relative_position) / 3e8
#         return cmath.exp(1j * LOS_phase) * math.sqrt(self.path_loss_normal) * (r_array_response * t_array_response.H)

#     def generate_array_response(self, transceiver, theta, fai):
#         """Generates array response vector based on antenna configuration."""
#         ant_type, ant_num = transceiver.ant_type, transceiver.ant_num
#         response = np.mat(np.ones((ant_num, 1), dtype=complex))
        
#         if ant_type == 'UPA':
#             row_num = int(math.sqrt(ant_num))
#             for i in range(row_num):
#                 for j in range(row_num):
#                     response[j + i * row_num, 0] = cmath.exp(1j * (math.sin(theta) * math.cos(fai) * i * math.pi + math.sin(theta) * math.sin(fai)))
#         elif ant_type == 'ULA':
#             for i in range(ant_num):
#                 response[i, 0] = cmath.exp(1j * math.sin(theta) * math.cos(fai) * i * math.pi)
#         return response if ant_type in ['UPA', 'ULA'] else np.mat(np.array([1]))

#     def update_CSI(self):
#         """Updates path loss and channel state information (CSI)."""
#         self.path_loss_normal = self.get_channel_path_loss()
#         self.path_loss_dB = normal_to_dB(self.path_loss_normal)
#         self.channel_matrix = self.get_estimated_channel_matrix()


import numpy as np
import math
import cmath
from math_tool import *

class mmWave_channel(object):
    """
    generate MmWave under UMi open
    input: distance, angle, pair entity object
    output: Instantaneous CSI
    """
    def __init__(self, transmitter, receiver, frequncy):
        """
        transmitter: object in entity.py
        receiver: object in entity.py
        
        """
        self.channel_name = ''
        self.n = 0
        self.sigma = 0
        self.transmitter = transmitter 
        self.receiver = receiver
        self.channel_type = self.init_type()    # 'UAV_RIS', 'UAV_user', 'UAV_attacker', 'RIS_user', 'RIS_attacker'
        
        # self.distance = np.linalg.norm(transmitter.coordinate - receiver.coordinate)
        self.frequncy = frequncy
        # init & updata path loss
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        # init & update channel CSI matrix
        self.channel_matrix = self.get_estimated_channel_matrix()
        
    def init_type(self):
        channel_type = self.transmitter.type+'_'+self.receiver.type
        if channel_type == 'UAV_RIS' or channel_type == 'RIS_UAV':
            self.n = 2.2
            self.sigma = 3
            self.channel_name = 'H_UR'
        elif channel_type == 'UAV_user' or channel_type == 'UAV_attacker':
            self.n = 3.5
            self.sigma = 3
            if channel_type =='UAV_user':
                self.channel_name = 'h_U_k,' + str(self.transmitter.index)
            elif channel_type == 'UAV_attacker':
                self.channel_name = 'h_U_p,' + str(self.transmitter.index)
        elif channel_type == 'user_UAV' or channel_type == 'attacker_UAV':
            self.n = 3.5
            self.sigma = 3
            if channel_type =='user_UAV':
                self.channel_name = 'h_U_k,' + str(self.transmitter.index)
            elif channel_type == 'attacker_UAV':
                self.channel_name = 'h_U_p,' + str(self.transmitter.index)
                
        elif channel_type == 'RIS_user' or channel_type == 'RIS_attacker':
            self.n = 2.8
            self.sigma = 3
            if channel_type =='RIS_user':
                self.channel_name = 'h_R_k,' + str(self.transmitter.index)
            elif channel_type == 'RIS_attacker':
                self.channel_name = 'h_R_p,' + str(self.transmitter.index)        
        elif channel_type == 'user_RIS' or channel_type == 'attacker_RIS':
            self.n = 2.8
            self.sigma = 3
            if channel_type =='user_RIS':
                self.channel_name = 'h_R_k,' + str(self.transmitter.index)
            elif channel_type == 'attacker_RIS':
                self.channel_name = 'h_R_p,' + str(self.transmitter.index)  
        return channel_type

    def get_channel_path_loss(self):
        """
        calculate the path loss including shadow fading 
        (in normal form)
        """
        distance = np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)
        PL = -20 * math.log10(4*math.pi/(3e8/self.frequncy)) - 10*self.n*math.log10(distance)
        shadow_loss = np.random.normal() * self.sigma
        # return dB_to_normal(PL - shadow_loss)
        return dB_to_normal(PL)

    def get_estimated_channel_matrix(self):
        """
        init & update channel matrix
        """
        # init matrix
        N_t = self.transmitter.ant_num
        N_r = self.receiver.ant_num
        channel_matrix = np.mat(np.ones(shape=(N_r,N_t),dtype=complex), dtype=complex)

        # get relevant coordinate receiver under transmitter system
        r_under_t_car_coor = get_coor_ref(\
        self.transmitter.coor_sys, \
        self.receiver.coordinate - self.transmitter.coordinate)
        # get relevant spherical_coordinate 
        r_t_r, r_t_theta, r_t_fai = cartesian_coordinate_to_spherical_coordinate(\
        cartesian_coordinate=r_under_t_car_coor\
        )

        # get relevant coordinate transmitter under receiver system
        t_under_r_car_coor = get_coor_ref(\
        #   remmber to Meet channel direction restrictions
        [-self.receiver.coor_sys[0], self.receiver.coor_sys[1], -self.receiver.coor_sys[2]],\
        self.transmitter.coordinate - self.receiver.coordinate)
        # get relevant spherical_coordinate 
        t_r_r, t_r_theta, t_r_fai = cartesian_coordinate_to_spherical_coordinate(\
        cartesian_coordinate=t_under_r_car_coor\
        )

        # calculate array response
        t_array_response = self.generate_array_response(self.transmitter, r_t_theta, r_t_fai)
        r_array_response = self.generate_array_response(self.receiver, t_r_theta, t_r_fai)
        array_response_product = r_array_response * t_array_response.H
        # get H_LOS
        #   get LOS path loss 
        PL = self.path_loss_normal
        
        #   get LOS phase shift
        LOS_fai = 2 * math.pi * self.frequncy * np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate) / 3e8
        channel_matrix = cmath.exp(1j*LOS_fai)* math.pow(PL, 0.5) * array_response_product
        
        return channel_matrix

    def generate_array_response(self, transceiver, theta, fai):
        """
        if the ant_type is 'UPA'
        generate_UPA_response
        if the ant_type is 'ULA'
        generate_ULA_response
        if the ant_type is 'single'
        generate_singleant_response
        """
        ant_type = transceiver.ant_type
        ant_num  = transceiver.ant_num

        if ant_type == 'UPA':
            row_num = int(math.sqrt(ant_num))
            Planar_response = np.mat(np.ones(shape=(ant_num, 1)), dtype=complex)
            for i in range(row_num):
                for j in range(row_num):
                    Planar_response[j+i*row_num,0] = cmath.exp(1j *\
                    (math.sin(theta) * math.cos(fai)*i*math.pi + math.sin(theta)*math.sin(fai))\
                    )
                
            return Planar_response
        elif ant_type == 'ULA':
            Linear_response = np.mat(np.ones(shape=(ant_num,1)), dtype=complex)
            for i in range(ant_num):
                Linear_response[i, 0] = cmath.exp(1j * math.sin(theta) * math.cos(fai)*i*math.pi)
            return Linear_response
        elif ant_type == 'single':
            return np.mat(np.array([1]))
        else:
            return False
        
    def update_CSI(self):
        """
        update pathloss and channel matrix
        """
        # init & updata path loss
        self.path_loss_normal = self.get_channel_path_loss()
        self.path_loss_dB = normal_to_dB(self.path_loss_normal)
        # init & update channel CSI matrix
        self.channel_matrix = self.get_estimated_channel_matrix()
