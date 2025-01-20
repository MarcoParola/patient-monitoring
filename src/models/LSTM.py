import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTM(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_dim=128, num_layers=1, dropout=0.4):
        super(LSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, x):
        # Add sequence length dimension (required for LSTM input)
        if x.dim() == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, seq_len=1, input_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)

        # Take the output from the last time step
        lstm_out_last = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Pass through the fully connected layer
        output = self.fc(lstm_out_last)  # (batch_size, output_dim)
        return output

