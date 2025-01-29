import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_mushroom_data(
    path="/home/emmanuel/Bureau/toy_bnn/mushroom_cleaned.csv",
    batch_size=64,
    test_size=0.2,
    random_state=42
):
    """
    Charge le dataset mushroom, le transforme en classification binaire
    Normalise et renvoie DataLoaders.
    """

    data = pd.read_csv(path)

    X = data.drop("class", axis=1).values
    y = data["class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, (X_train_tensor.shape[1])
