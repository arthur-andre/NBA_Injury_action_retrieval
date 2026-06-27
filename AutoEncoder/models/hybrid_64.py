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

        latent = torch.cat((pos_latent, angles_latent, vel_latent, lin_vel_latent), dim=2)
        #print(latent.shape)

        # Decode each feature independently
        pos_out = self.position_decoder(pos_latent)
        angles_out = self.angles_decoder(angles_latent)
        vel_out = self.vel_decoder(vel_latent)
        lin_vel_out = self.lin_vel_decoder(lin_vel_latent)
        
        return pos_out, angles_out, vel_out, lin_vel_out, latent
