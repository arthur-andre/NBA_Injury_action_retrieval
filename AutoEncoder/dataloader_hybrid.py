import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HybridDataset(Dataset):
    def __init__(self, json_file, frame_rate=60, clip_duration=2, max_missing=5, shift_step=10):
        self.json_file = json_file
        self.num_frames = frame_rate * clip_duration  # 120 frames for a 2-second clip at 60 FPS
        self.max_missing = max_missing
        self.shift_step = shift_step
        self.actions = self.load_action_data()

    def load_action_data(self):
        actions = {}
        with open(self.json_file) as f:
            data = json.load(f)
            players = data['action_data']['players']
            for player_id, player_data in players.items():
                frames = player_data['frames']
                valid_frames = [frame for frame in frames if frame is not None]
                if len(valid_frames) >= self.num_frames - self.max_missing:
                    actions[player_id] = valid_frames
        return actions

    def __len__(self):
        total_windows = 0
        for player_id, frames in self.actions.items():
            total_windows += max(0, (len(frames) - self.num_frames) // self.shift_step + 1)
        return total_windows

    def __getitem__(self, idx):
        current_idx = 0
        for player_id, frames in self.actions.items():
            num_windows = max(0, (len(frames) - self.num_frames) // self.shift_step + 1)
            if idx < current_idx + num_windows:
                window_start = (idx - current_idx) * self.shift_step
                window_frames = frames[window_start:window_start + self.num_frames]
                frame_numbers = [frame['frame_number'] for frame in window_frames]

                # Extract features for this window
                joint_angles_2d, joint_angles_3d, angular_velocity_2d, angular_velocity_3d, lin_vel_acc_3d, j3d_human = self.extract_features(window_frames)

                return (
                    torch.tensor(joint_angles_2d, dtype=torch.float32), 
                    torch.tensor(joint_angles_3d, dtype=torch.float32),
                    torch.tensor(angular_velocity_2d, dtype=torch.float32),
                    torch.tensor(angular_velocity_3d, dtype=torch.float32),
                    torch.tensor(lin_vel_acc_3d, dtype=torch.float32),
                    torch.tensor(j3d_human, dtype=torch.float32),
                    player_id
                )

            current_idx += num_windows
        raise IndexError("Index out of range")

    def extract_features(self, frames):
        joint_angles_2d = []
        joint_angles_3d = []
        angular_velocity_2d = []
        angular_velocity_3d = []
        lin_vel_acc_3d = []
        j3d_human = []

        for frame in frames:
            # Joint Angles (2D)
            joint_angles_2d.append([
                frame['joint_angles_2D']['right_leg']['ankle'],
                frame['joint_angles_2D']['right_leg']['knee'],
                frame['joint_angles_2D']['right_leg']['hip'],
                frame['joint_angles_2D']['left_leg']['ankle'],
                frame['joint_angles_2D']['left_leg']['knee'],
                frame['joint_angles_2D']['left_leg']['hip']
            ])

            # Joint Angles (3D)
            joint_angles_3d.append([
                frame['joint_angles_3D']['right_leg']['hip'],
                frame['joint_angles_3D']['right_leg']['knee'],
                frame['joint_angles_3D']['right_leg']['ankle'],
                frame['joint_angles_3D']['left_leg']['hip'],
                frame['joint_angles_3D']['left_leg']['knee'],
                frame['joint_angles_3D']['left_leg']['ankle']
            ])

            # Angular Velocity (2D)
            angular_velocity_2d.append([
                frame['angular_velocity_2D']['right_leg']['ankle'],
                frame['angular_velocity_2D']['right_leg']['knee'],
                frame['angular_velocity_2D']['right_leg']['hip'],
                frame['angular_velocity_2D']['left_leg']['ankle'],
                frame['angular_velocity_2D']['left_leg']['knee'],
                frame['angular_velocity_2D']['left_leg']['hip']
            ])

            # Angular Velocity (3D)
            angular_velocity_3d.append([
                frame['angular_velocity_3D']['right_leg']['hip'],
                frame['angular_velocity_3D']['right_leg']['knee'],
                frame['angular_velocity_3D']['right_leg']['ankle'],
                frame['angular_velocity_3D']['left_leg']['hip'],
                frame['angular_velocity_3D']['left_leg']['knee'],
                frame['angular_velocity_3D']['left_leg']['ankle']
            ])

            # Linear Velocities and Accelerations (3D)
            lin_vel_acc_3d.append([
                np.linalg.norm(frame['linear_velocity_3D']['right_leg']['knee']),
                np.linalg.norm(frame['linear_acceleration_3D']['right_leg']['knee']),
                np.linalg.norm(frame['linear_velocity_3D']['right_leg']['ankle']),
                np.linalg.norm(frame['linear_acceleration_3D']['right_leg']['ankle']),
                np.linalg.norm(frame['linear_velocity_3D']['left_leg']['knee']),
                np.linalg.norm(frame['linear_acceleration_3D']['left_leg']['knee']),
                np.linalg.norm(frame['linear_velocity_3D']['left_leg']['ankle']),
                np.linalg.norm(frame['linear_acceleration_3D']['left_leg']['ankle'])
            ])

            # 3D Human Joints
            j3d_human.append(np.array(frame['j3d_human']['coordinates']).flatten())

        # Convert lists to NumPy arrays
        joint_angles_2d = np.vstack(joint_angles_2d)
        joint_angles_3d = np.vstack(joint_angles_3d)
        angular_velocity_2d = np.vstack(angular_velocity_2d)
        angular_velocity_3d = np.vstack(angular_velocity_3d)
        lin_vel_acc_3d = np.vstack(lin_vel_acc_3d)
        j3d_human = np.vstack(j3d_human)

        return joint_angles_2d, joint_angles_3d, angular_velocity_2d, angular_velocity_3d, lin_vel_acc_3d, j3d_human
