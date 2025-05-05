# Import necessary modules
import numpy as np
from entity import *  # Importing UAV, RIS, Users, Attackers, and now Jammer
from channel import *  # mmWave channel modeling
from math_tool import *  # Math utilities

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from render import Render
from data_manager import DataManager

# Set random seed for reproducibility
np.random.seed(2)

class MiniSystem:
    """
    Defines the UAV-RIS communication system with:
    - UAV (Optimizing Beamforming)
    - RIS (Reflecting Intelligent Surface)
    - Users (Receiving Signal)
    - Attackers (Eavesdropping)
    - Jammers (Adding Interference)
    """

    def __init__(self, UAV_num=1, RIS_num=1, user_num=1, attacker_num=1, jammer_num=1,
                 fre=28e9, RIS_ant_num=16, UAV_ant_num=8, if_dir_link=1, if_with_RIS=True,
                 if_move_users=True, if_movements=True, reverse_x_y=(True, True), if_UAV_pos_state=True):
        """
        Initializes the system entities and environment.

        :param jammer_num: Number of jammers to add to the system.
        """
        self.if_dir_link = if_dir_link
        self.if_with_RIS = if_with_RIS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.jammer_num = jammer_num  # Adding jammers
        self.border = [(-25, 25), (0, 50)]

        # Data manager for storing system parameters
        self.data_manager = DataManager(file_path='./data',
                                        store_list=['beamforming_matrix', 'reflecting_coefficient', 'UAV_state',
                                                    'user_capacity', 'secure_capacity', 'attacker_capacity', 'G_power',
                                                    'reward', 'UAV_movement'])

        # 1. Initialize UAV
        self.UAV = UAV(
            coordinate=self.data_manager.read_init_location('UAV', 0),
            ant_num=UAV_ant_num,
            max_movement_per_time_slot=0.25
        )
        self.UAV.G = np.ones((self.UAV.ant_num, user_num), dtype=np.complex128)
        self.power_factor = 100
        self.UAV.G_Pmax = np.trace(self.UAV.G @ self.UAV.G.conj().T) * self.power_factor

        # 2. Initialize RIS
        self.RIS = RIS(
            coordinate=self.data_manager.read_init_location('RIS', 0),
            coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0),
            ant_num=RIS_ant_num
        )

        # 3. Initialize Users
        self.user_list = []
        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = -114
            self.user_list.append(user)

        # 4. Initialize Attackers
        self.attacker_list = []
        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros((user_num))
            attacker.noise_power = -114
            self.attacker_list.append(attacker)

        # 5. Initialize Jammers
        self.jammer_list = []
        for i in range(jammer_num):
            jammer_coordinate = self.data_manager.read_init_location('jammer', i)
            jammer = Jammer(coordinate=jammer_coordinate, index=i, power=5)
            self.jammer_list.append(jammer)

        # 6. Generate eavesdrop capacity array (P x K)
        self.eavesdrop_capacity_array = np.zeros((attacker_num, user_num))

        # 7. Initialize communication channels
        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)
        self.h_U_k = [mmWave_channel(user_k, self.UAV, fre) for user_k in self.user_list]
        self.h_R_k = [mmWave_channel(user_k, self.RIS, fre) for user_k in self.user_list]
        self.h_U_p = [mmWave_channel(attacker_p, self.UAV, fre) for attacker_p in self.attacker_list]
        self.h_R_p = [mmWave_channel(attacker_p, self.RIS, fre) for attacker_p in self.attacker_list]

        # 8. Update capacity
        self.update_channel_capacity()

        # 9. Render system visualization
        self.render_obj = Render(self)

    def reset(self):
        """
        Reset UAV, users, attackers, beamforming matrix, reflecting coefficient.
        Compatible with latest versions of libraries.
        """
        # 1. Reset UAV
        self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))
    
        # 2. Reset users
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)
    
        # 3. Reset attackers
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)
    
        # 4. Reset beamforming matrix
        self.UAV.G = np.ones((self.UAV.ant_num, self.user_num), dtype=np.complex128)
        self.UAV.G = np.asmatrix(self.UAV.G)  # Optional, depends on codebase
        self.UAV.G_Pmax = np.trace(self.UAV.G @ self.UAV.G.H) * self.power_factor
    
        # 5. Reset reflecting coefficient
        identity_matrix = np.eye(self.RIS.ant_num, dtype=np.complex128)
        self.RIS.Phi = np.asmatrix(np.diag(np.diag(identity_matrix)))
    
        # 6. Reset time index for rendering
        self.render_obj.t_index = 0
    
        # 7. Reset Channel State Information (CSI)
        self.H_UR.update_CSI()
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
    
        # 8. Reset capacity
        self.update_channel_capacity()

    def step(self, action_0=0, action_1=0, G=0, Phi=0, set_pos_x=0, set_pos_y=0):
        """
        Test step: only move UAV and update channel.
        """
        # 0. Update time index for rendering
        self.render_obj.t_index += 1
    
        # 1. Update entity positions
        if self.if_move_users:
            self.user_list[0].update_coordinate(0.2, -0.5 * math.pi)
            self.user_list[1].update_coordinate(0.2, -0.5 * math.pi)
    
        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot
    
            if self.reverse_x_y[0]:
                move_x = -move_x
            if self.reverse_x_y[1]:
                move_y = -move_y
    
            self.UAV.coordinate[0] += move_x
            self.UAV.coordinate[1] += move_y
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV.coordinate[0] = set_pos_x
            self.UAV.coordinate[1] = set_pos_y
    
        # 2. Update channel CSI
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
    
        # Disable direct links if needed
        if not self.if_dir_link:
            for h in self.h_U_k + self.h_U_p:
                h.channel_matrix = np.zeros_like(h.channel_matrix, dtype=complex)
    
        # RIS inclusion/exclusion logic
        if not self.if_with_RIS:
            self.H_UR.channel_matrix = np.zeros((self.RIS.ant_num, self.UAV.ant_num), dtype=complex)
        else:
            self.H_UR.update_CSI()
    
        # 3. Update beamforming matrix & reflecting phase shift
        self.UAV.G = convert_list_to_complex_matrix(G, (self.UAV.ant_num, self.user_num)) * np.sqrt(self.power_factor)
    
        if self.if_with_RIS:
            self.RIS.Phi = convert_list_to_complex_diag(Phi, self.RIS.ant_num)
    
        # 4. Update channel capacity
        self.update_channel_capacity()
    
        # 5. Store current system state
        self.store_current_system_sate()
    
        # 6. Get new observation state
        new_state = self.observe()
    
        # 7. Calculate reward
        reward = math.tanh(self.reward())
    
        # 8. Check boundary conditions
        done = False
        x, y = self.UAV.coordinate[:2]
    
        if not (self.border[0][0] <= x <= self.border[0][1]) or not (self.border[1][0] <= y <= self.border[1][1]):
            done = True
            reward = -10
    
        self.data_manager.store_data([reward], 'reward')
    
        return new_state, reward, done, []

    def reward(self):
        """
        Computes the reward at the current step.
        Rewards depend on beamforming power and secure capacity vs. eavesdropper capacity.
        """
        reward = 0.0
        penalty = 0.0
    
        # Compute current transmission power
        G_matrix = np.array(self.UAV.G)
        power_used = np.trace(G_matrix @ G_matrix.conj().T).real
    
        # Penalize if power exceeds max allowed
        if abs(power_used) > abs(self.UAV.G_Pmax):
            reward = (abs(self.UAV.G_Pmax) - abs(power_used)) / self.power_factor
        else:
            for user in self.user_list:
                # Calculate difference between capacity and eavesdropper max capacity
                secrecy_margin = user.capacity - np.max(self.eavesdrop_capacity_array[:, user.index])
    
                if secrecy_margin < user.QoS_constrain:
                    penalty += secrecy_margin - user.QoS_constrain
                else:
                    reward += secrecy_margin / (self.user_num * 2)
    
            if penalty < 0:
                reward = penalty/100 * self.user_num * 10
    
        return reward
    
    def observe(self):
        """
        Used in function main to get current state.
        The state is a list with:
        - Real and imaginary parts of the comprehensive channel for users and attackers
        - UAV position (if enabled)
        """
        comprehensive_channel_elements_list = []
    
        for entity in self.user_list + self.attacker_list:
            # Flatten the comprehensive_channel array safely and convert to real + imag parts
            flattened = np.ravel(entity.comprehensive_channel)
            real_part = np.real(flattened).tolist()
            imag_part = np.imag(flattened).tolist()
            comprehensive_channel_elements_list.extend(real_part + imag_part)
    
        UAV_position_list = []
        if self.if_UAV_pos_state:
            UAV_position_list = list(self.UAV.coordinate)
    
        return comprehensive_channel_elements_list + UAV_position_list

    def store_current_system_sate(self):
        """
        Stores the current system state, including beamforming matrix, 
        reflecting coefficients, UAV position, capacities, and power usage.
        """
        # 1. Store beamforming matrix (flattened real & imaginary parts)
        G_flat = self.UAV.G.flatten()
        self.data_manager.store_data(G_flat.tolist(), 'beamforming_matrix')
    
        # 2. Store reflecting coefficient matrix (diagonal values only)
        if self.RIS.Phi is not None:
            diag_elements = np.diag(self.RIS.Phi)
            self.data_manager.store_data(diag_elements.tolist(), 'reflecting_coefficient')
        else:
            self.data_manager.store_data([], 'reflecting_coefficient')
    
        # 3. Store UAV state (x, y position)
        self.data_manager.store_data(list(map(float, self.UAV.coordinate[:2])), 'UAV_state')
    
        # 4. Store user capacity (secure and normal)
        user_capacities = [user.secure_capacity for user in self.user_list] + \
                          [user.capacity for user in self.user_list]
        self.data_manager.store_data(user_capacities, 'user_capacities_combined')
    
        # 5. Store G power info (transmit power and power max)
        transmit_power = float(np.trace(self.UAV.G @ self.UAV.G.conj().T).real)
        G_power_info = [transmit_power, self.UAV.G_Pmax]
        self.data_manager.store_data(G_power_info, 'G_power')
    
        # 6. Store individual user capacity
        user_caps = [user.capacity for user in self.user_list]
        self.data_manager.store_data(user_caps, 'user_capacity')
    
        # 7. Store attacker capacities
        attacker_caps = [attacker.capacity for attacker in self.attacker_list]
        self.data_manager.store_data(attacker_caps, 'attacker_capacity')
    
        # 8. Store secure capacities
        secure_caps = [user.secure_capacity for user in self.user_list]
        self.data_manager.store_data(secure_caps, 'secure_capacity')

    
    def update_channel_capacity(self):
        """
        Function to calculate user and attackers' capacity.
        """
        # 1. Calculate eavesdrop rate
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
            attacker.comprehensive_channel = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
        
        # 2. Calculate unsecure rate
        for user in self.user_list:
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            # 3. Calculate secure rate
            user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """
        Calculates the comprehensive_channel of attacker p.
        """
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).conj().T @ self.H_UR.channel_matrix
        return h_U_p.conj().T + Psi.conj().T @ H_c

    def calculate_comprehensive_channel_of_user_k(self, k):
        """
        Calculates the comprehensive_channel of user k.
        """
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).conj().T @ self.H_UR.channel_matrix
        return h_U_k.conj().T + Psi.conj().T @ H_c

    def calculate_capacity_of_user_k(self, k):
        """
        Calculates the capacity of a user.
        """     
        noise_power = self.user_list[k].noise_power
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).conj().T @ self.H_UR.channel_matrix
        G_k = self.UAV.G[:, k]
        
        G_k_ = np.zeros((self.UAV.ant_num, 1), dtype=complex) if len(self.user_list) == 1 else np.hstack((self.UAV.G[:, :k], self.UAV.G[:, k+1:]))
        
        alpha_k = np.abs((h_U_k.conj().T + Psi.conj().T @ H_c) @ G_k) ** 2
        beta_k = np.linalg.norm((h_U_k.conj().T + Psi.conj().T @ H_c) @ G_k_) ** 2 + dB_to_normal(noise_power) * 1e-3
        
        return np.log10(1 + np.abs(alpha_k / beta_k))

    def calculate_capacity_array_of_attacker_p(self, p):
        """
        Calculates the attacker's capacities to K users.
        """
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).conj().T @ self.H_UR.channel_matrix
        
        if K == 1:
            G_k = self.UAV.G
            G_k_ = np.zeros((self.UAV.ant_num, 1), dtype=complex)
            alpha_p = np.abs((h_U_p.conj().T + Psi.conj().T @ H_c) @ G_k) ** 2
            beta_p = np.linalg.norm((h_U_p.conj().T + Psi.conj().T @ H_c) @ G_k_) ** 2 + dB_to_normal(noise_power) * 1e-3
            return np.array([np.log10(1 + np.abs(alpha_p / beta_p))])
        else:
            result = np.zeros(K)
            for k in range(K):
                G_k = self.UAV.G[:, k]
                G_k_ = np.hstack((self.UAV.G[:, :k], self.UAV.G[:, k+1:]))
                alpha_p = np.abs((h_U_p.conj().T + Psi.conj().T @ H_c) @ G_k) ** 2
                beta_p = np.linalg.norm((h_U_p.conj().T + Psi.conj().T @ H_c) @ G_k_) ** 2 + dB_to_normal(noise_power) * 1e-3
                result[k] = np.log10(1 + np.abs(alpha_p / beta_p))
            return result

    def calculate_secure_capacity_of_user_k(self, k=2):
        """
        Calculates the secure rate of user k.
        """
        user = self.user_list[k]
        R_k_unsecure = user.capacity
        R_k_maxeavesdrop = max(self.eavesdrop_capacity_array[:, k])
        return max(0, R_k_unsecure - R_k_maxeavesdrop)

    def calculate_capacity_of_user_k(self, k):
        """
        Calculates the communication capacity for user k with jamming power added.

        :param k: User index
        :return: User's secure communication capacity
        """
        noise_power = self.user_list[k].noise_power
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = np.diag(self.RIS.Phi)
        H_c = np.diagflat(h_R_k).conj().T @ self.H_UR.channel_matrix
        G_k = self.UAV.G[:, k]

        # Compute jamming power
        Power_jamming = sum(j.power for j in self.jammer_list)

        alpha_k = np.abs((h_U_k.conj().T + Psi.conj().T @ H_c) @ G_k) ** 2
        beta_k = np.linalg.norm((h_U_k.conj().T + Psi.conj().T @ H_c) @ self.UAV.G[:, :k]) ** 2 + dB_to_normal(noise_power) * 1e-3 + Power_jamming

        return np.log10(1 + abs(alpha_k / beta_k))

    def calculate_secure_capacity_of_user_k(self, k):
        """
        Calculates secure capacity of user k considering jamming interference.
        """
        user = self.user_list[k]
        R_k_unsecure = user.capacity
        R_k_maxeavesdrop = np.max(self.eavesdrop_capacity_array[:, k])
        return max(0, R_k_unsecure - R_k_maxeavesdrop)

    def get_system_action_dim(self):
        """
        Function used to get the dimension of actions in the system.
        """
        result = 0
        # 1. UAV movement (2D movement)
        result += 2
        
        # 2. RIS reflecting elements (if applicable)
        if getattr(self, 'if_with_RIS', False):  # Use getattr to avoid attribute errors
            result += getattr(self.RIS, 'ant_num', 0)  # Ensure RIS has 'ant_num'
    
        # 3. Beamforming matrix dimension
        result += 2 * getattr(self.UAV, 'ant_num', 1) * getattr(self, 'user_num', 1)
        
        return result
    
    
    def get_system_state_dim(self):
        """
        Function used to get the dimension of states in the system.
        """
        result = 0
        # 1. Users' and attackers' comprehensive channel
        result += 2 * (getattr(self, 'user_num', 1) + getattr(self, 'attacker_num', 0)) * getattr(self.UAV, 'ant_num', 1)
    
        # 2. UAV position state (if applicable)
        if getattr(self, 'if_UAV_pos_state', False):  
            result += 3  # Assuming 3D coordinates for UAV position
    
        return result
    
