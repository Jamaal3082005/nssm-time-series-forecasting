import torch
import torch.nn as nn

class NSSM(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim

        self.transition = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.observation = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_next = self.transition(z)
        x_hat = self.observation(z_next)
        return z_next, x_hat
