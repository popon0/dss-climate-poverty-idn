"""
lstm_model.py

LSTM model for forecasting emission & poverty time series.

Structure:
- LSTMForecast: 2-input LSTM with 2-layer stacked recurrent units
- train_lstm_model: training utility function
- evaluate_lstm_model: evaluation with MAE & R² metrics
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score


class LSTMForecast(nn.Module):
    """
    LSTM-based forecaster for emission & poverty.

    - Input: sequence of shape (batch, seq_len, 2)
    - Output: next timestep prediction (emission_scaled, poverty_scaled)

    Args:
        input_size (int): number of input features (default=2)
        hidden_size (int): hidden layer dimension
        num_layers (int): number of LSTM layers
        dropout (float): dropout between LSTM layers
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): input tensor (batch, seq_len, input_size)

        Returns:
            torch.Tensor: prediction (batch, input_size), scaled [0–1]
        """
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        return torch.sigmoid(self.fc(out[:, -1, :]))


def train_lstm_model(
    model: LSTMForecast,
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
    epochs: int = 2000,
    lr: float = 0.005,
    batch_size: int = 64,
    verbose: bool = True,
) -> LSTMForecast:
    """
    Train an LSTM model with MSE loss and Adam optimizer.

    Args:
        model (LSTMForecast): initialized model
        X_seq (torch.Tensor): input sequences (N, seq_len, 2)
        y_seq (torch.Tensor): target sequences (N, 2)
        epochs (int): number of training epochs
        lr (float): learning rate
        batch_size (int): batch size
        verbose (bool): print training loss every 100 epochs

    Returns:
        LSTMForecast: trained model
    """
    dataset = TensorDataset(X_seq, y_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose and epoch % 100 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"[Epoch {epoch:04d}] Loss = {avg_loss:.6f}")

    return model


def evaluate_lstm_model(
    model: LSTMForecast,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    """
    Evaluate LSTM model using MAE and R² scores.

    Args:
        model (LSTMForecast): trained model
        X_val (torch.Tensor): validation inputs (N, seq_len, 2)
        y_val (torch.Tensor): validation targets (N, 2)

    Returns:
        dict[str, float]: metrics
            - mae_emisi
            - mae_kemiskinan
            - r2_emisi
            - r2_kemiskinan
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).numpy()
        y_true = y_val.numpy()

    return {
        "mae_emisi": mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
        "mae_kemiskinan": mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
        "r2_emisi": r2_score(y_true[:, 0], y_pred[:, 0]),
        "r2_kemiskinan": r2_score(y_true[:, 1], y_pred[:, 1]),
    }
