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

        def update_channel_capacity(self):
        """
        function used in step to calculate user and attackers' capacity 
        """
        # 1 calculate eavesdrop rate
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
            # remmeber to update comprehensive_channel
            attacker.comprehensive_channel = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
        # 2 calculate unsecure rate
        for user in self.user_list:
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            # 3 calculate secure rate
            user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            # remmeber to update comprehensive_channel
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)

        def calculate_comprehensive_channel_of_attacker_p(self, p):
            """
            used in update_channel_capacity to calculate the comprehensive_channel of attacker p
            """
            h_U_p = self.h_U_p[p].channel_matrix
            h_R_p = self.h_R_p[p].channel_matrix
            Psi = diag_to_vector(self.RIS.Phi)
            H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
            return h_U_p.H + Psi.H * H_c
    
        def calculate_comprehensive_channel_of_user_k(self, k):
            """
            used in update_channel_capacity to calculate the comprehensive_channel of user k
            """
            h_U_k = self.h_U_k[k].channel_matrix
            h_R_k = self.h_R_k[k].channel_matrix
            Psi = diag_to_vector(self.RIS.Phi)
            H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
            return h_U_k.H + Psi.H * H_c
    
        def calculate_capacity_of_user_k(self, k):
            """
            function used in update_channel_capacity to calculate one user
            """     
            noise_power = self.user_list[k].noise_power
            h_U_k = self.h_U_k[k].channel_matrix
            h_R_k = self.h_R_k[k].channel_matrix
            Psi = diag_to_vector(self.RIS.Phi)
            H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
            G_k = self.UAV.G[:, k]
            G_k_ = 0
            if len(self.user_list) == 1:
                G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
            else:
                G_k_1 = self.UAV.G[:, 0:k]
                G_k_2 = self.UAV.G[:, k+1:]
                G_k_ = np.hstack((G_k_1, G_k_2))
            alpha_k = math.pow(abs((h_U_k.H + Psi.H * H_c) * G_k), 2)
            beta_k = math.pow(np.linalg.norm((h_U_k.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
            return math.log10(1 + abs(alpha_k / beta_k))
    
        def calculate_capacity_array_of_attacker_p(self, p):
            """
            function used in update_channel_capacity to calculate one attacker capacities to K users
            output is a K length np.array ,shape: (K,)
            """
            K = len(self.user_list)
            noise_power = self.attacker_list[p].noise_power
            h_U_p = self.h_U_p[p].channel_matrix
            h_R_p = self.h_R_p[p].channel_matrix
            Psi = diag_to_vector(self.RIS.Phi)
            H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
            if K == 1:
                G_k = self.UAV.G
                G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
                alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
                beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
                return np.array([math.log10(1 + abs(alpha_p / beta_p))])
            else:
                result = np.zeros(K)
                for k in range(K):
                    G_k = G_k = self.UAV.G[:, k]
                    G_k_1 = self.UAV.G[:, 0:k]
                    G_k_2 = self.UAV.G[:, k+1:]
                    G_k_ = np.hstack((G_k_1, G_k_2))
                    alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
                    beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
                    result[k] = math.log10(1 + abs(alpha_p / beta_p))
                return result
    
        def calculate_secure_capacity_of_user_k(self, k=2):
            """
            function used in update_channel_capacity to calculate the secure rate of user k
            """
            user = self.user_list[k]
            R_k_unsecure = user.capacity
            R_k_maxeavesdrop = max(self.eavesdrop_capacity_array[:, k])
            secrecy_rate= max(0, R_k_unsecure - R_k_maxeavesdrop)
            return secrecy_rate
    
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
        H_c = np.diag(h_R_k).conj().T @ self.H_UR.channel_matrix
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
