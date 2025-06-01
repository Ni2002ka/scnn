import torch
import torch.nn as nn
from utils.data import mse, unscale_log_return


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, hidden_layers=1, output_size=1, dropout=0.1):
        super(MLPModel, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, scale_min, scale_max, epochs=50, lr=0.001, batch_size=32, huber_delta=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    model = MLPModel(input_size).to(device)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        y_pred_train = model(X_train_tensor).cpu().squeeze().numpy()
        y_pred_test = model(X_test_tensor).cpu().squeeze().numpy()

    y_pred_train_unscaled = unscale_log_return(scale_min, scale_max, y_pred_train)
    y_pred_test_unscaled = unscale_log_return(scale_min, scale_max, y_pred_test)
    y_train_unscaled = unscale_log_return(scale_min, scale_max, y_train)
    y_test_unscaled = unscale_log_return(scale_min, scale_max, y_test)

    train_error = mse(y_pred_train, y_train)
    test_error = mse(y_pred_test, y_test)

    return train_error, test_error, model, (y_train_unscaled, y_pred_train_unscaled), (y_test_unscaled, y_pred_test_unscaled)
