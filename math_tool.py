import math
import cmath
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def cartesian_coordinate_to_spherical_coordinate(cartesian_coord):
    """
    Converts Cartesian coordinates to spherical coordinates.
    Input:  1x3 np.array -> [x, y, z]
    Output: 1x3 np.array -> [r, theta, phi]
    """
    r = np.linalg.norm(cartesian_coord)
    theta = math.atan(np.linalg.norm(cartesian_coord[:2]) / (cartesian_coord[2] if abs(cartesian_coord[2]) >= 1e-8 else 1e-8))

    x, y = cartesian_coord[0], cartesian_coord[1]
    x = x if abs(x) >= 1e-8 else 1e-8
    y = y if abs(y) >= 1e-8 else 1e-8

    if x > 0:
        phi = math.atan(y / x)
    elif x < 0 and y >= 0:
        phi = math.atan(y / x) + math.pi
    else:
        phi = math.atan(y / x) - (math.pi if x < 0 else 0)

    return np.array([r, theta, phi])

def get_coor_ref(coor_sys, coor):
    """
    input:  coor_sys: normalized 1,3 np.array list (1,3)
            coor: coordinate under earth system
    output: referenced coordinate for x,y, normalized 1,3 np.array
    """
    x_ref = np.dot(coor_sys[0],coor)
    y_ref = np.dot(coor_sys[1],coor)
    z_ref = np.dot(coor_sys[2],coor)
    return np.array([x_ref, y_ref, z_ref])

def normalize_vector(vector):
    """Returns a normalized vector."""
    return vector / np.linalg.norm(vector)

def transform_coordinates(coor_sys, coor):
    """
    Transforms coordinates to a reference system.
    Input:
        coor_sys: 3x3 np.array (normalized)
        coor: Coordinate in the original system
    Output:
        Transformed coordinate as a 1x3 np.array
    """
    return np.array([np.dot(axis, coor) for axis in coor_sys])

def dB_to_normal(dB):
    """
    input: dB
    output: normal vaule
    """
    return math.pow(10, (dB/10))

def normal_to_dB(normal):
    """
    input: normal
    output: dB value
    """
    return -10 * math.log10(normal)

def diag_to_vector(diagonal_matrix):
    """Converts a diagonal matrix into a vector."""
    return np.diag(diagonal_matrix).reshape(-1, 1)

def vector_to_diag(vector):
    """Converts a vector into a diagonal matrix."""
    return np.diagflat(vector)

def ensure_positive(value):
    """Returns max(0, value)."""
    return max(0, value)

def dataframe_to_dict(df):
    """Converts a Pandas DataFrame to a dictionary."""
    return {col: df[col].values for col in df.columns}

def list_to_complex_matrix(real_list, shape):
    """
    Converts a list to a complex matrix.
    Input:
        real_list: 2 * (N*K) list
        shape: (N, K) tuple
    Output:
        N x K complex matrix
    """
    N, K = shape
    matrix = np.zeros((N, K), dtype=complex)
    for i in range(N):
        for j in range(K):
            matrix[i, j] = real_list[2 * (i * K + j)] + 1j * real_list[2 * (i * K + j) + 1]
    return np.mat(matrix)

def list_to_complex_diag(real_list, size):
    """
    Converts a list into a complex diagonal matrix.
    Input:
        real_list: List of size M
        size: Number of diagonal elements (M)
    Output:
        M x M complex diagonal matrix
    """
    diag_matrix = np.zeros((size, size), dtype=complex)
    np.fill_diagonal(diag_matrix, [cmath.exp(1j * value * math.pi) for value in real_list])
    return np.mat(diag_matrix)

def map_range(x, x_range, y_range):
    """
    Maps a value from one range to another.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)
