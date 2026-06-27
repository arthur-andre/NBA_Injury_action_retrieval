import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np

test_features_nan = []

class PlayerDatasetAllPlayers(Dataset):
    def __init__(self, json_file, frame_rate=60, clip_duration=2, max_missing=5, shift_step=10):
        """
        json_file: Path to the single JSON file.
        frame_rate: Frames per second of the dataset (default is 60 FPS).
        clip_duration: Duration of each clip in seconds (e.g., 2 seconds).
        max_missing: Max allowed missing frames within a clip.
        shift_step: Step size for sliding the window (in number of frames).
        """
        self.json_file = json_file
        self.num_frames = frame_rate * clip_duration  # 120 frames for a 2-second clip at 60 FPS
        self.max_missing = max_missing
        self.shift_step = shift_step
        self.actions = self.load_action_data()

    def load_action_data(self):
        """
        Load action data from the provided JSON file and extract valid frames for all players.
        """
        actions = {}  # Dictionary to store frames for each player
        with open(self.json_file) as f:
            data = json.load(f)
            players = data['action_data']['players']
            for player_id, player_data in players.items():
                frames = player_data['frames']
                # Filter frames with missing data (if any)
                valid_frames = [frame for frame in frames if frame is not None]
                if len(valid_frames) >= self.num_frames - self.max_missing:
                    actions[player_id] = valid_frames
        return actions

    def __len__(self):
        """
        Return the number of sliding windows we can extract for all players.
        """
        total_windows = 0
        for player_id, frames in self.actions.items():
            total_windows += max(0, (len(frames) - self.num_frames) // self.shift_step + 1)
        return total_windows

    def __getitem__(self, idx):
        """
        Get a specific time window (2-second clip) from the dataset for all players.
        """
        current_idx = 0
        for player_id, frames in self.actions.items():
            num_windows = max(0, (len(frames) - self.num_frames) // self.shift_step + 1)
            if idx < current_idx + num_windows:
                # Find the exact window within the action
                window_start = (idx - current_idx) * self.shift_step
                window_frames = frames[window_start:window_start + self.num_frames]

                # Get frame numbers for tracking
                frame_numbers = [frame['frame_number'] for frame in window_frames]

                # Convert the frames to a flat feature vector
                features = self.extract_features(window_frames, player_id, frame_numbers)

                # Return features and frame numbers for tracking, including player_id
                return torch.tensor(features, dtype=torch.float32), frame_numbers, player_id

            current_idx += num_windows
        raise IndexError("Index out of range")

    def extract_features(self, frames, player_id, frame_numbers):
        """
        Flatten and combine the features for all frames in a window.
        Also checks for NaN/Inf values in individual features before concatenating.
        """
        pose_features = []
        for i, frame in enumerate(frames):
            # Extract and flatten features
            joint_angles_2d = [
                frame['joint_angles_2D']['right_leg']['ankle'],
                frame['joint_angles_2D']['right_leg']['knee'],
                frame['joint_angles_2D']['right_leg']['hip'],
                frame['joint_angles_2D']['left_leg']['ankle'],
                frame['joint_angles_2D']['left_leg']['knee'],
                frame['joint_angles_2D']['left_leg']['hip']
            ]
            angular_velocity_2d = [
                frame['angular_velocity_2D']['right_leg']['ankle'],
                frame['angular_velocity_2D']['right_leg']['knee'],
                frame['angular_velocity_2D']['right_leg']['hip'],
                frame['angular_velocity_2D']['left_leg']['ankle'],
                frame['angular_velocity_2D']['left_leg']['knee'],
                frame['angular_velocity_2D']['left_leg']['hip']
            ]
            joint_angles_3d = [
                frame['joint_angles_3D']['right_leg']['hip'],
                frame['joint_angles_3D']['right_leg']['knee'],
                frame['joint_angles_3D']['right_leg']['ankle'],
                frame['joint_angles_3D']['left_leg']['hip'],
                frame['joint_angles_3D']['left_leg']['knee'],
                frame['joint_angles_3D']['left_leg']['ankle']
            ]
            angular_velocity_3d = [
                frame['angular_velocity_3D']['right_leg']['hip'],
                frame['angular_velocity_3D']['right_leg']['knee'],
                frame['angular_velocity_3D']['right_leg']['ankle'],
                frame['angular_velocity_3D']['left_leg']['hip'],
                frame['angular_velocity_3D']['left_leg']['knee'],
                frame['angular_velocity_3D']['left_leg']['ankle']
            ]
            lin_vel_3d_knee_right = np.linalg.norm(frame['linear_velocity_3D']['right_leg']['knee'])
            lin_acc_3d_knee_right = np.linalg.norm(frame['linear_acceleration_3D']['right_leg']['knee'])
            lin_vel_3d_ankle_right = np.linalg.norm(frame['linear_velocity_3D']['right_leg']['ankle'])
            lin_acc_3d_ankle_right = np.linalg.norm(frame['linear_acceleration_3D']['right_leg']['ankle'])

            lin_vel_3d_knee_left = np.linalg.norm(frame['linear_velocity_3D']['left_leg']['knee'])
            lin_acc_3d_knee_left = np.linalg.norm(frame['linear_acceleration_3D']['left_leg']['knee'])
            lin_vel_3d_ankle_left = np.linalg.norm(frame['linear_velocity_3D']['left_leg']['ankle'])
            lin_acc_3d_ankle_left = np.linalg.norm(frame['linear_acceleration_3D']['left_leg']['ankle'])

            j3d_human = np.array(frame['j3d_human']['coordinates']).flatten()

            # Check for NaNs or Infs in individual features
            self.check_nan_inf(np.array(joint_angles_2d), player_id, frame['frame_number'], "joint_angles_2d")
            self.check_nan_inf(np.array(angular_velocity_2d), player_id, frame['frame_number'], "angular_velocity_2d")
            self.check_nan_inf(np.array(joint_angles_3d), player_id, frame['frame_number'], "joint_angles_3d")
            self.check_nan_inf(np.array(angular_velocity_3d), player_id, frame['frame_number'], "angular_velocity_3d")
            self.check_nan_inf(j3d_human, player_id, frame['frame_number'], "j3d_human")

            # Combine the features
            feature_vector = (
                joint_angles_2d +
                angular_velocity_2d +
                joint_angles_3d +
                angular_velocity_3d +
                [lin_vel_3d_knee_right] +
                [lin_acc_3d_knee_right] +
                [lin_vel_3d_ankle_right] +
                [lin_acc_3d_ankle_right] +
                [lin_vel_3d_knee_left] +
                [lin_acc_3d_knee_left] +
                [lin_vel_3d_ankle_left] +
                [lin_acc_3d_ankle_left] +
                list(j3d_human)
            )
            pose_features.append(feature_vector)

        # Flatten the features for all frames into a single vector
        flattened_features = np.concatenate(pose_features)

        return flattened_features

    def check_nan_inf(self, features, player_id, frame_numbers, feature_name):
        """
        Check for NaN or Inf values in features and report the player, frame, and problematic feature.
        """
        if np.isnan(features).any():
            nan_idx = np.where(np.isnan(features))[0]
            print(f"NaN detected for player {player_id} in frames {frame_numbers} for feature {feature_name}. NaN in features at index: {nan_idx}")
            if feature_name not in test_features_nan:
                test_features_nan.append(feature_name)
        if np.isinf(features).any():
            inf_idx = np.where(np.isinf(features))[0]
            print(f"Inf detected for player {player_id} in frames {frame_numbers} for feature {feature_name}. Inf in features at index: {inf_idx}")

# Functions to load JSON files and create dataloaders
def load_all_json_files(root_folder):
    """
    Load all player_data.json files from all subdirectories under the root folder.
    """
    json_files = []
    print(f"Loading JSON files from: {root_folder}")
    for subdir in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, subdir)
        json_file_path = os.path.join(folder_path, 'players_data.json')
        if os.path.exists(json_file_path):
            json_files.append(json_file_path)
    return json