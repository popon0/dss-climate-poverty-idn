# Copyright 2025 Teuku Hafiez Ramadhan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
anfis_model.py

Implementation of an Adaptive Neuro-Fuzzy Inference System (ANFIS)
with Gaussian membership functions and a first-order Sugeno model.

Structure:
- GaussianMF: Gaussian membership function module
- ANFIS: Neuro-fuzzy network (2 inputs: emission, poverty)
- train_anfis_model: training utility function
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GaussianMF(nn.Module):
    """
    Gaussian membership function:
        μ(x) = exp(-0.5 * ((x - μ) / σ)^2)

    Args:
        mu (float): initial mean
        sigma (float): initial standard deviation
    """

    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)


class ANFIS(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System (Sugeno-1).

    - Two inputs: emission_scaled, poverty_scaled
    - Three Gaussian MFs per input (low, medium, high)
    - 3x3 = 9 fuzzy rules
    - Linear output layer combines rule firing strengths
    """

    def __init__(self):
        super().__init__()
        # Membership functions
        self.mf_emisi = nn.ModuleList(
            [GaussianMF(0.2, 0.05), GaussianMF(0.5, 0.05), GaussianMF(0.8, 0.05)]
        )
        self.mf_kemiskinan = nn.ModuleList(
            [GaussianMF(0.2, 0.05), GaussianMF(0.5, 0.05), GaussianMF(0.8, 0.05)]
        )

        # Consequent linear function (Sugeno layer)
        self.linear = nn.Linear(9, 1)

        # Initialize weights for stability
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            self.linear.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e, k = x[:, 0], x[:, 1]

        fe = torch.stack([mf(e) for mf in self.mf_emisi], dim=1)  # (N, 3)
        fk = torch.stack([mf(k) for mf in self.mf_kemiskinan], dim=1)  # (N, 3)

        # Rule firing strengths: outer product → (N, 3, 3) → flatten → (N, 9)
        rules = torch.bmm(fe.unsqueeze(2), fk.unsqueeze(1)).view(-1, 9)

        return self.linear(rules)


def train_anfis_model(
    model: ANFIS,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 2000,
    lr: float = 0.005,
    batch_size: int = 64,
    verbose: bool = True,
) -> ANFIS:
    """
    Train an ANFIS model with MSE loss and Adam optimizer.

    Args:
        model (ANFIS): initialized ANFIS model
        X (torch.Tensor): input features (N, 2)
        y (torch.Tensor): target values (N, 1)
        epochs (int): training epochs
        lr (float): learning rate
        batch_size (int): batch size for training
        verbose (bool): if True, print loss every 100 epochs

    Returns:
        ANFIS: trained model
    """
    dataset = TensorDataset(X, y)
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
