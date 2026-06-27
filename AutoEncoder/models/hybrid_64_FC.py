import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_64 import FeatureEncoder, PositionEncoder, PositionDecoder, FeatureDecoderLSTM

class HybridAutoencoder(nn.Module):
    def __init__(self, pos_shape=(32, 3, 120, 20), angles_shape=(32, 120, 12), vel_shape=(32, 120, 12), lin_vel_shape=(32, 120, 8), latent_dim=64):
        super(HybridAutoencoder, self).__init__()
        
        self.fc_reduce_pos = nn.Linear(64, 16)
        self.fc_reduce_angles = nn.Linear(64, 16)
        self.fc_reduce_vel = nn.Linear(64, 16)
        self.fc_reduce_lin_vel = nn.Linear(64, 16)
        
        self.fc_expand_pos = nn.Linear(16, 64)
        self.fc_expand_angles = nn.Linear(16, 64)
        self.fc_expand_vel = nn.Linear(16, 64)
        self.fc_expand_lin_vel = nn.Linear(16, 64)

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

        pos_latent_reduced = self.fc_reduce_pos(pos_latent)
        angles_latent_reduced = self.fc_reduce_angles(angles_latent)
        vel_latent_reduced = self.fc_reduce_vel(vel_latent)
        lin_vel_latent_reduced = self.fc_reduce_lin_vel(lin_vel_latent)

        latent = torch.cat((pos_latent_reduced, angles_latent_reduced, vel_latent_reduced, lin_vel_latent_reduced), dim=2)
        
        pos_latent_expanded = self.fc_expand_pos(pos_latent_reduced)
        angles_latent_expanded = self.fc_expand_angles(angles_latent_reduced)
        vel_latent_expanded = self.fc_expand_vel(vel_latent_reduced)
        lin_vel_latent_expanded = self.fc_expand_lin_vel(lin_vel_latent_reduced)
        
        
        # Decode each feature independently
        pos_out = self.position_decoder(pos_latent_expanded)
        angles_out = self.angles_decoder(angles_latent_expanded)
        vel_out = self.vel_decoder(vel_latent_expanded)
        lin_vel_out = self.lin_vel_decoder(lin_vel_latent_expanded)
        
        return pos_out, angles_out, vel_out, lin_vel_out, latent