import torch
import torch.nn as nn
from src.nssm_model import NSSM

def train_nssm(train_data, epochs=100):
    model = NSSM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    z = torch.zeros(1, model.latent_dim)

    for epoch in range(epochs):
        total_loss = 0
        for x in train_data:
            optimizer.zero_grad()
            z, x_hat = model(z)
            loss = loss_fn(x_hat.squeeze(), torch.tensor(x))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_data):.4f}")
    return model
