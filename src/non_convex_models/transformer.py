import torch
import torch.nn as nn
from utils.data import mse, unscale_log_return


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.1, output_size=1):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_projection(x)  # Shape: (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use last time step
        return self.output_layer(x)


def train_transformer(X_train, y_train, X_test, y_test, scale_min, scale_max, epochs=50, lr=0.001, batch_size=32, huber_delta=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]
    model = TransformerModel(input_size).to(device)

    # criterion = nn.MSELoss()
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

    # Unscale predictions
    y_pred_train_unscaled = unscale_log_return(scale_min, scale_max, y_pred_train)
    y_pred_test_unscaled = unscale_log_return(scale_min, scale_max, y_pred_test)
    y_train_unscaled = unscale_log_return(scale_min, scale_max, y_train)
    y_test_unscaled = unscale_log_return(scale_min, scale_max, y_test)

    train_error = mse(y_pred_train, y_train)
    test_error = mse(y_pred_test, y_test)

    return train_error, test_error, model, (y_train_unscaled, y_pred_train_unscaled), (y_test_unscaled, y_pred_test_unscaled)
