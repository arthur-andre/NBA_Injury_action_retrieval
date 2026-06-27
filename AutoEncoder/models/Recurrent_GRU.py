import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder for Angles, Velocities, Linear Velocities using GRU
class FeatureEncoderGRU(nn.Module):
    def __init__(self, input_size, latent_dim=64):
        super(FeatureEncoderGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=latent_dim, num_layers=2, batch_first=True)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        x, h_n = self.gru(x)  # GRU only returns the hidden state (no cell state like LSTM)
        return x


# Position Encoder with 3D Convolutional layers followed by GRUs
class PositionEncoderGRU(nn.Module):
    def __init__(self, input_channels=3, num_vertices=20, seq_len=120, latent_dim=64):
        super(PositionEncoderGRU, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1)
        
        # GRU to capture temporal dependencies
        self.gru = nn.GRU(input_size=128 * num_vertices, hidden_size=latent_dim, num_layers=2, batch_first=True)
        
    def forward(self, x):
        # Initial shape: [batch_size, 3, 120, 20]
        x = F.relu(self.conv1(x.unsqueeze(2)))  # Add 1 dimension for Conv3D
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten for GRU: [batch_size, seq_len, conv_features]
        x = x.view(x.size(0), x.size(3), -1)  # Reshape to [batch_size, 120, 128 * 20 = 2560]
        x, h_n = self.gru(x)  # Pass through GRU

        return x


# GRU-based Decoders for Angles, Velocities, Linear Velocities
class FeatureDecoderGRU(nn.Module):
    def __init__(self, latent_dim, output_size, hidden_size=64):
        super(FeatureDecoderGRU, self).__init__()
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Decode using GRU
        x, h_n = self.gru(x)
        x = self.output_layer(x)
        return x


# GRU-based Hybrid Autoencoder
class GRUAutoencoder(nn.Module):
    def __init__(self, pos_shape=(32, 3, 120, 20), angles_shape=(32, 120, 12), vel_shape=(32, 120, 12), lin_vel_shape=(32, 120, 8), latent_dim=64):
        super(GRUAutoencoder, self).__init__()
        
        # GRU Encoders
        self.position_encoder = PositionEncoderGRU(input_channels=pos_shape[1], num_vertices=pos_shape[3], latent_dim=latent_dim)
        self.angles_encoder = FeatureEncoderGRU(input_size=angles_shape[2], latent_dim=latent_dim)
        self.vel_encoder = FeatureEncoderGRU(input_size=vel_shape[2], latent_dim=latent_dim)
        self.lin_vel_encoder = FeatureEncoderGRU(input_size=lin_vel_shape[2], latent_dim=latent_dim)
        
        # GRU Decoders
        self.position_decoder = PositionDecoder(latent_dim=latent_dim, num_vertices=pos_shape[3], output_channels=pos_shape[1])
        self.angles_decoder = FeatureDecoderGRU(latent_dim=latent_dim, output_size=angles_shape[2])
        self.vel_decoder = FeatureDecoderGRU(latent_dim=latent_dim, output_size=vel_shape[2])
        self.lin_vel_decoder = FeatureDecoderGRU(latent_dim=latent_dim, output_size=lin_vel_shape[2])
    
    def forward(self, pos, angles, vel, lin_vel):
        # Encode
        pos_latent = self.position_encoder(pos)
        angles_latent = self.angles_encoder(angles)
        vel_latent = self.vel_encoder(vel)
        lin_vel_latent = self.lin_vel_encoder(lin_vel)

        # Concatenate latent representations
        latent = torch.cat((pos_latent, angles_latent, vel_latent, lin_vel_latent), dim=2)

        # Decode each feature independently
        pos_out = self.position_decoder(pos_latent)
        angles_out = self.angles_decoder(angles_latent)
        vel_out = self.vel_decoder(vel_latent)
        lin_vel_out = self.lin_vel_decoder(lin_vel_latent)
        
        return pos_out, angles_out, vel_out, lin_vel_out, latent
