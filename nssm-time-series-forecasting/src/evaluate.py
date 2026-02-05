import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_data):
    z = torch.zeros(1, model.latent_dim)
    preds = []

    for x in test_data:
        z, x_hat = model(z)
        preds.append(x_hat.item())

    rmse = np.sqrt(mean_squared_error(test_data, preds))
    mae = mean_absolute_error(test_data, preds)
    return rmse, mae
