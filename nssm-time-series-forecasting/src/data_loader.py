import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_time_series(n_steps=6000):
    t = np.arange(n_steps)
    trend = 0.01 * t
    seasonality = np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 0.1, n_steps)
    return trend + seasonality + noise

def prepare_data():
    data = generate_time_series()
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))

    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]

    return train, val, test
