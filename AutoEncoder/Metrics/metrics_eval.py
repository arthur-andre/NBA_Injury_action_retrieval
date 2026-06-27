import numpy as np
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


######################### METRICS #########################

def mpjpe(predicted, ground_truth):
    """
    Computes the Mean Per Joint Position Error (MPJPE).
    
    Parameters:
    - predicted: np.array of shape (N, J, 3), where N is the number of samples (e.g., frames),
                 J is the number of joints, and 3 represents the (x, y, z) coordinates.
    - ground_truth: np.array of shape (N, J, 3), the ground truth 3D joint positions.

    Returns:
    - mpjpe_value: float, the average MPJPE over all samples.
    """
    
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth data"
    
    error_per_joint = np.linalg.norm(predicted - ground_truth, axis=-1)  # Shape: (N, J)
    
    mean_error_per_sample = np.mean(error_per_joint, axis=1)  # Shape: (N,)
    
    mpjpe_value = np.mean(mean_error_per_sample)
    
    return mpjpe_value

def mean_absolute_angular_error(predicted, ground_truth):
    """
    Computes the Mean Absolute Angular Error (MAAE) for joint angles.
    
    Parameters:
    - predicted: np.array of shape (N, J), where N is the number of samples (e.g., frames),
                 J is the number of joints with angle values (in radians or degrees).
    - ground_truth: np.array of shape (N, J), the ground truth joint angles.
    
    Returns:
    - maae_value: float, the average MAAE over all samples.
    """
    
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth data"
    
    absolute_error_per_joint = np.abs(predicted - ground_truth)  # Shape: (N, J)
    
    mean_error_per_sample = np.mean(absolute_error_per_joint, axis=1)  # Shape: (N,)
    
    maae_value = np.mean(mean_error_per_sample)
    
    return maae_value

def rmse(predicted, ground_truth):
    """
    Computes the Root Mean Squared Error (RMSE) for velocities or accelerations.
    
    Parameters:
    - predicted: np.array of shape (N, J), where N is the number of samples (e.g., frames),
                 J is the number of joints with velocity/acceleration values.
    - ground_truth: np.array of shape (N, J), the ground truth velocity/acceleration values.
    
    Returns:
    - rmse_value: float, the average RMSE over all samples.
    """
    
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth data"
    
    squared_error_per_joint = (predicted - ground_truth) ** 2  # Shape: (N, J)
    
    mean_squared_error_per_sample = np.mean(squared_error_per_joint, axis=1)  # Shape: (N,)
    
    rmse_value = np.sqrt(np.mean(mean_squared_error_per_sample))
    
    return rmse_value

def cosine_similarity(predicted, ground_truth):
    """
    Computes the average cosine similarity between predicted and ground truth velocity/acceleration vectors.
    
    Parameters:
    - predicted: np.array of shape (N, J, 3), where N is the number of samples (e.g., frames),
                 J is the number of joints with 3D velocity/acceleration vectors.
    - ground_truth: np.array of shape (N, J, 3), the ground truth velocity/acceleration vectors.
    
    Returns:
    - avg_cosine_similarity: float, the average cosine similarity over all samples.
    """
    
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth data"
    
    cosine_sim_per_sample = [
        1 - cosine(predicted[i].flatten(), ground_truth[i].flatten())
        for i in range(predicted.shape[0])
    ]
    
    avg_cosine_similarity = np.mean(cosine_sim_per_sample)
    
    return avg_cosine_similarity

def dynamic_time_warping(predicted, ground_truth):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences of velocities or accelerations.
    
    Parameters:
    - predicted: np.array of shape (N, J), where N is the number of samples (e.g., frames),
                 J is the number of joints with velocity/acceleration values.
    - ground_truth: np.array of shape (N, J), the ground truth velocity/acceleration values.
    
    Returns:
    - dtw_distance: float, the DTW distance between the sequences.
    """
    
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth data"
    
    dtw_distance, _ = fastdtw(predicted, ground_truth, dist=euclidean)
    
    return dtw_distance


######################### VIZUALIZATION #########################

def compare_angles(angles_1, angles_2):
    """
    Compare two sets of angles.
    
    Parameters:
    - angles_1: np.array of shape (N, J), angles from first data set (e.g., in radians or degrees).
    - angles_2: np.array of shape (N, J), angles from second data set.
    
    Returns:
    - average_angle_diff: float, the mean difference in angles.
    """
    angle_diff = np.abs(angles_1 - angles_2)  # Compute absolute differences
    return np.mean(angle_diff)

def compare_angular_velocities(ang_vel_1, ang_vel_2):
    """
    Compare angular velocities between two datasets.
    
    Parameters:
    - ang_vel_1: np.array of shape (N, J), angular velocities from first data set.
    - ang_vel_2: np.array of shape (N, J), angular velocities from second data set.
    
    Returns:
    - average_ang_vel_diff: float, the mean difference in angular velocities.
    """
    ang_vel_diff = np.linalg.norm(ang_vel_1 - ang_vel_2, axis=-1)  # Euclidean distance per frame
    return np.mean(ang_vel_diff)


def compare_linear_quantities(lin_1, lin_2):
    """
    Compare linear velocities or accelerations between two datasets.
    
    Parameters:
    - lin_1: np.array of shape (N, J, 3), linear velocities/accelerations from first dataset (xyz coordinates).
    - lin_2: np.array of shape (N, J, 3), linear velocities/accelerations from second dataset.
    
    Returns:
    - average_lin_diff: float, the mean difference in linear velocities/accelerations.
    """
    lin_diff = np.linalg.norm(lin_1 - lin_2, axis=-1)  # Euclidean distance for each joint
    return np.mean(lin_diff)


def compare_datasets(angles_1, angles_2, ang_vel_1, ang_vel_2, lin_vel_1, lin_vel_2, lin_acc_1, lin_acc_2):
    """
    Compare all features (angles, angular velocities, linear velocities, and accelerations) between two datasets.
    
    Parameters:
    - angles_1, angles_2: np.arrays of shape (N, J), angles in both datasets.
    - ang_vel_1, ang_vel_2: np.arrays of shape (N, J), angular velocities.
    - lin_vel_1, lin_vel_2: np.arrays of shape (N, J, 3), linear velocities (xyz coordinates).
    - lin_acc_1, lin_acc_2: np.arrays of shape (N, J, 3), linear accelerations (xyz coordinates).
    
    Returns:
    - comparison_results: dict, with average differences for each feature.
    """
    # Compare angles
    angle_diff = compare_angles(angles_1, angles_2)
    
    # Compare angular velocities
    ang_vel_diff = compare_angular_velocities(ang_vel_1, ang_vel_2)
    
    # Compare linear velocities
    lin_vel_diff = compare_linear_quantities(lin_vel_1, lin_vel_2)
    
    # Compare linear accelerations
    lin_acc_diff = compare_linear_quantities(lin_acc_1, lin_acc_2)
    
    # Return a dictionary of results
    comparison_results = {
        'angle_difference': angle_diff,
        'angular_velocity_difference': ang_vel_diff,
        'linear_velocity_difference': lin_vel_diff,
        'linear_acceleration_difference': lin_acc_diff
    }
    
    return comparison_results
