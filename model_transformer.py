import torch
import torch.nn as nn

class LSTMTransformer(nn.Module):
    def __init__(self, input_size, lstm_hidden=128, trans_heads=4, trans_layers=2, num_classes=3):
        super(LSTMTransformer, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=trans_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=trans_layers
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        x, _ = self.lstm(x)         # → (B, T, H)
        x = self.transformer(x)     # → (B, T, H)
        x = x[:, -1, :]             # 마지막 timestep
        out = self.fc(x)            # → (B, C)
        return out
