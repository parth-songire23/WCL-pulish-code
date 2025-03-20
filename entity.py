import numpy as np
import math

class UAV(object):
    """
    UAV (Unmanned Aerial Vehicle) with:
    - Position (Coordinate)
    - Beamforming Matrix
    - Antennas
    - Movement Capability
    """
    def __init__(self, coordinate, index=0, rotation=0, ant_num=16, ant_type='ULA', max_movement_per_time_slot=0.5):
        self.max_movement_per_time_slot = max_movement_per_time_slot
        self.type = 'UAV'
        self.coordinate = coordinate
        self.rotation = rotation
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]

        # Initialize beamforming matrix (Power allocation)
        self.G = np.mat(np.zeros((ant_num, 1)))
        self.G_Pmax = 0

    def reset(self, coordinate):
        """ Reset UAV position """
        self.coordinate = coordinate

    def move(self, distance_delta_d, direction_fai, delta_angle=0):
        """ Move UAV in 2D space """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

class RIS(object):
    """
    Reconfigurable Intelligent Surface (RIS) with:
    - Reflecting elements
    - Position (Coordinate)
    """
    def __init__(self, coordinate, coor_sys_z, index=0, ant_num=36, ant_type='UPA'):
        self.type = 'RIS'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index

        coor_sys_z = coor_sys_z / np.linalg.norm(coor_sys_z)
        coor_sys_x = np.cross(coor_sys_z, np.array([0, 0, 1]))
        coor_sys_x = coor_sys_x / np.linalg.norm(coor_sys_x)
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = [coor_sys_x, coor_sys_y, coor_sys_z]

        # Initialize reflecting phase shift
        self.Phi = np.mat(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)

class User(object):
    """
    User (Receiver) with:
    - Position
    - Single Antenna
    - Capacity (Throughput)
    """
    def __init__(self, coordinate, index, ant_num=1, ant_type='single'):
        self.type = 'user'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # Communication capacity
        self.capacity = 0
        self.secure_capacity = 0
        self.QoS_constrain = 0

        # Signal processing properties
        self.comprehensive_channel = 0
        self.noise_power = -114  # Noise floor (dBm)

    def reset(self, coordinate):
        """ Reset user position """
        self.coordinate = coordinate

class Attacker(object):
    """
    Attacker (Eavesdropper) with:
    - Position
    - Single Antenna
    - Eavesdropping Capacity
    """
    def __init__(self, coordinate, index, ant_num=1, ant_type='single'):
        self.type = 'attacker'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # Attack capacity (per user)
        self.capacity = 0
        self.comprehensive_channel = 0
        self.noise_power = -114  # Noise floor

    def reset(self, coordinate):
        """ Reset attacker position """
        self.coordinate = coordinate

class Jammer(object):
    """
    Jammer with:
    - Position (Coordinate)
    - Interference Power
    - Single Antenna
    """
    def __init__(self, coordinate, index, power=5, ant_num=1, ant_type='single'):
        self.type = 'jammer'
        self.coordinate = coordinate  # Position in space
        self.index = index
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.power = power  # Jamming power (Watts)
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    def reset(self, coordinate):
        """ Reset jammer position """
        self.coordinate = coordinate

    def generate_interference(self, signal):
        """
        Add jamming interference to a given signal.
        The interference is modeled as additive white Gaussian noise (AWGN).
        """
        interference = self.power * np.random.normal(0, 1, signal.shape)
        return signal + interference
