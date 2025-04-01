from entity import UAV, RIS
from channel import mmWave_channel
import numpy as np

# Initialize UAV at a specific position (x, y, z)
uav_position = np.array([0.0001, 0.001, 25])  # UAV at 25m altitude
UAV_test = UAV(coordinate=uav_position)

# Initialize RIS at a specific position with a given orientation
ris_position = np.array([25, 0.001, 25])  # RIS at same height as UAV
ris_normal_vector = np.array([-1, 0.0001, 0.0001])  # Surface normal direction
RIS_test = RIS(coordinate=ris_position, coor_sys_z=ris_normal_vector)

# Initialize mmWave channel between UAV and RIS
test_channel = mmWave_channel(UAV_test, RIS_test, frequency=28e9)  # Fixed typo in "frequncy" to "frequency"

# Print Channel Properties for Verification
print("\n📡 **UAV-to-RIS mmWave Channel Initialized**")
print(f"➡️ Channel Type: {test_channel.channel_type}")
print(f"🔹 Path Loss (Normal): {test_channel.path_loss_normal:.6f}")
print(f"🔹 Path Loss (dB): {test_channel.path_loss_dB:.2f} dB")
print(f"🔹 Channel Matrix Shape: {test_channel.channel_matrix.shape}")
print(f"🔹 First Entry of Channel Matrix: {test_channel.channel_matrix[0, 0]}")
