import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_64 import FeatureEncoder, PositionEncoder, PositionDecoder, FeatureDecoderLSTM


class DenoisingAutoencoder(nn.Module):
    def __init__(self, pos_shape=(32, 3, 120, 20), angles_shape=(32, 120, 12), vel_shape=(32, 120, 12), lin_vel_shape=(32, 120, 8), latent_dim=64, noise_std=0.1):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoders
        self.position_encoder = PositionEncoder(input_channels=pos_shape[1], num_vertices=pos_shape[3], latent_dim=latent_dim)
        self.angles_encoder = FeatureEncoder(input_size=angles_shape[2], latent_dim=latent_dim)
        self.vel_encoder = FeatureEncoder(input_size=vel_shape[2], latent_dim=latent_dim)
        self.lin_vel_encoder = FeatureEncoder(input_size=lin_vel_shape[2], latent_dim=latent_dim)
        
        # Decoders
        self.position_decoder = PositionDecoder(latent_dim=latent_dim, num_vertices=pos_shape[3], output_channels=pos_shape[1])
        self.angles_decoder = FeatureDecoderLSTM(latent_dim=latent_dim, output_size=angles_shape[2])
        self.vel_decoder = FeatureDecoderLSTM(latent_dim=latent_dim, output_size=vel_shape[2])
        self.lin_vel_decoder = FeatureDecoderLSTM(latent_dim=latent_dim, output_size=lin_vel_shape[2])
        
        # Noise level for denoising
        self.noise_std = noise_std
    
    def add_noise(self, data):
        """Adds Gaussian noise to the data."""
        noise = torch.randn_like(data) * self.noise_std
        return data + noise
    
    def forward(self, pos, angles, vel, lin_vel):
        # Add noise to inputs
        noisy_pos = self.add_noise(pos)
        noisy_angles = self.add_noise(angles)
        noisy_vel = self.add_noise(vel)
        noisy_lin_vel = self.add_noise(lin_vel)
        
        # Encode noisy inputs
        pos_latent = self.position_encoder(noisy_pos)
        angles_latent = self.angles_encoder(noisy_angles)
        vel_latent = self.vel_encoder(noisy_vel)
        lin_vel_latent = self.lin_vel_encoder(noisy_lin_vel)

        # Concatenate latent representations
        latent = torch.cat((pos_latent, angles_latent, vel_latent, lin_vel_latent), dim=2)

        # Decode each feature independently
        pos_out = self.position_decoder(pos_latent)
        angles_out = self.angles_decoder(angles_latent)
        vel_out = self.vel_decoder(vel_latent)
        lin_vel_out = self.lin_vel_decoder(lin_vel_latent)
        
        return pos_out, angles_out, vel_out, lin_vel_out, latent
