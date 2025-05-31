import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.data import unscale_log_return, mse


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)
        return out

def train_lstm(X_train, y_train, X_test, y_test, scale_min, scale_max, epochs=50, lr=0.001, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]
    model = LSTMModel(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
        y_pred_train = model(X_train_tensor).cpu().squeeze().numpy()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_test = model(X_test_tensor).cpu().squeeze().numpy()

    # Unscale
    y_pred_train_unscaled = unscale_log_return(scale_min, scale_max, y_pred_train)
    y_pred_test_unscaled = unscale_log_return(scale_min, scale_max, y_pred_test)
    y_train_unscaled = unscale_log_return(scale_min, scale_max, y_train)
    y_test_unscaled = unscale_log_return(scale_min, scale_max, y_test)

    return mse(y_pred_train, y_train), mse(y_pred_test, y_test), model, (y_train_unscaled, y_pred_train_unscaled), (y_test_unscaled, y_pred_test_unscaled)
