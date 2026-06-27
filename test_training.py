import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import torch.nn as nn


class HybridDataset(Dataset):
    def __init__(self, root_dir, frame_rate=60, clip_duration=2, max_missing=5, shift_step=10, device='cpu'):
        self.frame_rate = frame_rate
        self.clip_duration = clip_duration
        self.num_frames = frame_rate * clip_duration  # 120 frames for a 2-second clip at 60 FPS
        self.max_missing = max_missing
        self.shift_step = shift_step
        self.root_dir = root_dir
        self.device = device
        
        # List to store information about each window
        self.window_info = []
        
        # Set to keep track of printed folders to avoid redundant prints
        self.printed_folders = set()
        
        # Load all data and prepare window_info
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Scans through all subdirectories to find players_data.json files,
        loads the data, and prepares window_info list with window metadata.
        """
        print("Scanning directories for players_data.json files...")
        folder_count = 0  # To count how many folders have been processed
        # Walk through all subdirectories
        for subdir, _, files in os.walk(self.root_dir):
            if 'players_data.json' in files:
                json_path = os.path.join(subdir, 'players_data.json')
                folder_name = os.path.basename(subdir)
                folder_count += 1
                print(f"Processing folder {folder_name} ({folder_count})...")  # Immediate feedback on folder progress
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue  # Skip this file if there's an error
                
                players = data.get('action_data', {}).get('players', {})
                for player_id, player_data in players.items():
                    frames = player_data.get('frames', [])
                    valid_frames = [frame for frame in frames if frame is not None]
                    
                    if len(valid_frames) >= self.num_frames - self.max_missing:
                        num_windows = max(0, (len(valid_frames) - self.num_frames) // self.shift_step + 1)
                        
                        for window_num in range(num_windows):
                            window_start = window_num * self.shift_step
                            window_frames = valid_frames[window_start : window_start + self.num_frames]
                            
                            # Append window metadata to window_info
                            self.window_info.append({
                                'folder': folder_name,
                                'player_id': player_id,
                                'frames': window_frames
                            })
        
        print(f"Total windows collected: {len(self.window_info)}")

    def __len__(self):
        return len(self.window_info)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        window = self.window_info[idx]
        folder = window['folder']
        player_id = window['player_id']
        frames = window['frames']
        
        # Print the folder being processed if not already printed
        if folder not in self.printed_folders:
            print(f"Fetching data from folder: {folder}")
            self.printed_folders.add(folder)
        
        # Extract features for this window
        features = self.extract_features(frames)
        if features is None:
            raise ValueError(f"Failed to extract features for index {idx}")
        
        joint_angles_2d, joint_angles_3d, angular_velocity_2d, angular_velocity_3d, lin_vel_acc_3d, j3d_human = features
        
        # Concatenate joint angles (2D and 3D) and velocities (2D and 3D)
        angles = np.hstack((joint_angles_2d, joint_angles_3d))
        velocities = np.hstack((angular_velocity_2d, angular_velocity_3d))
        
        # Reshape j3d_human from (num_frames, 105) to (3, num_frames, 35)
        j3d_human = j3d_human.reshape(self.num_frames, -1, 3).transpose(2, 0, 1)  # (num_frames, 105) -> (3, num_frames, 35)
        
        return (
            torch.tensor(j3d_human, dtype=torch.float32).to(self.device),       # Positional data (3D joints with separate channels for x, y, z)
            torch.tensor(angles, dtype=torch.float32).to(self.device),          # Concatenated angles (2D + 3D)
            torch.tensor(velocities, dtype=torch.float32).to(self.device),      # Concatenated velocities (2D + 3D)
            torch.tensor(lin_vel_acc_3d, dtype=torch.float32).to(self.device),  # Linear velocities and accelerations
            player_id                                           # Player ID
        )
    
    def extract_features(self, frames):
        """
        Extracts required features from the frames.
        
        Args:
            frames (list): List of frame dictionaries.
        
        Returns:
            tuple: Extracted features as NumPy arrays.
        """
        joint_angles_2d = []
        joint_angles_3d = []
        angular_velocity_2d = []
        angular_velocity_3d = []
        lin_vel_acc_3d = []
        j3d_human = []
        
        # Indices to keep from the 35 keypoints
        keypoint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 34, 13, 14, 15, 10, 11, 12, 21, 22, 30, 31]
        
        for frame in frames:
            try:
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

                # 3D Human Joints (extract x, y, z separately and filter by indices)
                j3d_human_full = np.array(frame['j3d_human']['coordinates']).reshape(35, 3)
                j3d_human_filtered = j3d_human_full[keypoint_indices, :].flatten()  # Keep only the specified keypoints
                j3d_human.append(j3d_human_filtered)
            except KeyError as e:
                print(f"Key error during feature extraction: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error during feature extraction: {e}")
                return None
        
        # Convert lists to NumPy arrays
        joint_angles_2d = np.vstack(joint_angles_2d)
        joint_angles_3d = np.vstack(joint_angles_3d)
        angular_velocity_2d = np.vstack(angular_velocity_2d)
        angular_velocity_3d = np.vstack(angular_velocity_3d)
        lin_vel_acc_3d = np.vstack(lin_vel_acc_3d)
        j3d_human = np.vstack(j3d_human)
        
        return joint_angles_2d, joint_angles_3d, angular_velocity_2d, angular_velocity_3d, lin_vel_acc_3d, j3d_human


import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder for Angles, Velocities, Linear Velocities using LSTM
class FeatureEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=64):
        super(FeatureEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=latent_dim, num_layers=2, batch_first=True)  # Removed bidirectional
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        x, (h_n, c_n) = self.lstm(x)
        
        return x


# Position Encoder with 3D Convolutional layers followed by LSTMs
class PositionEncoder(nn.Module):
    def __init__(self, input_channels=3, num_vertices=20, seq_len=120, latent_dim=64):
        super(PositionEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        
        # LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=128 * num_vertices, hidden_size=latent_dim, num_layers=2, batch_first=True)  # Removed bidirectional
        
    def forward(self, x):
        # Initial shape: [batch_size, 3, 120, 20]
        x = F.relu(self.conv1(x.unsqueeze(2)))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        
        # Flatten for LSTM: [batch_size, seq_len, conv_features]
        x = x.view(x.size(0), x.size(3), -1)  # Reshape to [batch_size, 120, 128 * 20 = 2560]

        x, (h_n, c_n) = self.lstm(x)  # Pass through LSTM

        
        return x

# Position Decoder with Deconvolutions and LSTM
class PositionDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_vertices=20, output_channels=3):
        super(PositionDecoder, self).__init__()

        # LSTM to reverse the temporal encoding
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=128 * num_vertices, num_layers=2, batch_first=True)  # Removed bidirectional
        
        # Conv1D layer to project the LSTM output to the required input for ConvTranspose3D
        self.conv1d = nn.Conv1d(in_channels=120, out_channels=128 * num_vertices, kernel_size=1)
        
        # Deconvolution layers to reverse the Conv3D process
        self.deconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.deconv1 = nn.ConvTranspose3d(in_channels=32, out_channels=output_channels, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        # x shape: [batch_size, 120, hidden_dim]
        x, (h_n, c_n) = self.lstm(x)  # LSTM output
        
        # Reshape for Conv1D: swap seq_len and hidden_dim
        # x = x.transpose(1, 2)  # Change shape to [batch_size, hidden_dim, seq_len]
        # x = self.conv1d(x)  # Project to spatial dimensions for deconv layers
        # print("After Conv1D:", x.shape)  # Should match the expected ConvTranspose input shape

        # Reshape to match input to deconvolutions: [batch_size, 128, 1, 120, 20]
        x = x.view(x.size(0), 120, 1, 128, 20)
        #transposed 1 and 3
        x = x.permute(0, 3, 2, 1, 4)
        
        # Pass through deconvolution layers (reverse of encoder)
        x = F.relu(self.deconv3(x))

        x = F.relu(self.deconv2(x))

        x = self.deconv1(x)  # Output shape should match original input shape: [batch_size, 3, 1, 120, 20]

        
        return x.squeeze(2)  # Remove the extra dimension to get [batch_size, 3, 120, 20]


# LSTM-based Decoders for Angles, Velocities, Linear Velocities
class FeatureDecoderLSTM(nn.Module):
    def __init__(self, latent_dim, output_size, hidden_size=64):
        super(FeatureDecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)  # Removed bidirectional
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Decode using LSTM
        x, (h_n, c_n) = self.lstm(x)  # Use the full sequence output, no need to repeat across time steps

        x = self.output_layer(x)  # Output layer processes each timestep
        
        return x


# Hybrid Autoencoder with Independent Decoders
class HybridAutoencoder(nn.Module):
    def __init__(self, pos_shape=(32, 3, 120, 20), angles_shape=(32, 120, 12), vel_shape=(32, 120, 12), lin_vel_shape=(32, 120, 8), latent_dim=64):
        super(HybridAutoencoder, self).__init__()
        
        # Encoders
        self.position_encoder = PositionEncoder(input_channels=pos_shape[1], num_vertices=pos_shape[3], latent_dim=latent_dim)
        self.angles_encoder = FeatureEncoder(input_size=angles_shape[2], latent_dim=latent_dim)
        self.vel_encoder = FeatureEncoder(input_size=vel_shape[2], latent_dim=latent_dim)
        self.lin_vel_encoder = FeatureEncoder(input_size=lin_vel_shape[2], latent_dim=latent_dim)
        
        # Decoders for each feature type
        self.position_decoder = PositionDecoder(latent_dim=latent_dim, num_vertices=pos_shape[3], output_channels=pos_shape[1])
        self.angles_decoder = FeatureDecoderLSTM(latent_dim=latent_dim, output_size=angles_shape[2])
        self.vel_decoder = FeatureDecoderLSTM(latent_dim=latent_dim, output_size=vel_shape[2])
        self.lin_vel_decoder = FeatureDecoderLSTM(latent_dim=latent_dim, output_size=lin_vel_shape[2])
    
    def forward(self, pos, angles, vel, lin_vel):
        # Encode each feature
        pos_latent = self.position_encoder(pos)
        angles_latent = self.angles_encoder(angles)
        vel_latent = self.vel_encoder(vel)
        lin_vel_latent = self.lin_vel_encoder(lin_vel)

        # Decode each feature independently
        pos_out = self.position_decoder(pos_latent)
        angles_out = self.angles_decoder(angles_latent)
        vel_out = self.vel_decoder(vel_latent)
        lin_vel_out = self.lin_vel_decoder(lin_vel_latent)
        
        return pos_out, angles_out, vel_out, lin_vel_out


def main():
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create dataset with GPU support
    dataset = HybridDataset(root_dir='/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/CHI_NYK', device=device)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = HybridAutoencoder().to(device)  # Move model to GPU
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Set the model to training mode
    model.train()

    # Define the number of epochs
    num_epochs = 20

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0  # Track total loss over the epoch

        for pos, angles, velocities, lin_vel, _ in dataloader:
            # Move input tensors to the same device as the model
            pos = pos.to(device)
            angles = angles.to(device)
            velocities = velocities.to(device)
            lin_vel = lin_vel.to(device)

            # Forward pass
            pos_out, angles_out, vel_out, lin_vel_out = model(pos, angles, velocities, lin_vel)

            # Compute the reconstruction loss for each feature type
            loss_pos = loss_fn(pos_out, pos)
            loss_angles = loss_fn(angles_out, angles)
            loss_vel = loss_fn(vel_out, velocities)
            loss_lin_vel = loss_fn(lin_vel_out, lin_vel)

            # Total loss is the sum of individual losses
            loss = loss_pos + loss_angles + loss_vel + loss_lin_vel

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Print the average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}')

    print("Training complete.")

if __name__ == "__main__":
    main()