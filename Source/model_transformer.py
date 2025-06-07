import torch
import torch.nn as nn

class MultiHeadTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        seq_len=30,
        num_classes_direction=3,   # (상승/관망/하락)
        num_classes_position=5,    # (포지션 분류 예시)
        num_candle_outputs=4       # (OHLC 예측용: open, high, low, close)
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === 각 Task별 Head 정의 ===
        self.head_direction = nn.Linear(d_model, num_classes_direction)
        self.head_position = nn.Linear(d_model, num_classes_position)
        self.head_return = nn.Linear(d_model, 1)
        self.head_candle = nn.Linear(d_model, num_candle_outputs)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_linear(x)                # (batch, seq_len, d_model)
        x = x + self.pos_encoder[:, :self.seq_len]
        x = self.transformer(x)                 # (batch, seq_len, d_model)
        x_last = x[:, -1, :]                    # (batch, d_model) → 마지막 timestep만 사용

        out = {
            "direction": self.head_direction(x_last),    # (batch, num_classes_direction)
            "position": self.head_position(x_last),      # (batch, num_classes_position)
            "return": self.head_return(x_last).squeeze(-1),  # (batch,) 회귀
            "candle": self.head_candle(x_last),          # (batch, 4) (OHLC)
        }
        return out

    def get_loss(self, outputs, y_dict, loss_weights=None):
        """
        outputs: dict (direction, position, return, candle)
        y_dict: dict (direction, position, return, candle)
        loss_weights: dict (각 loss 가중치, 없으면 전부 1)
        """
        loss_fns = {
            "direction": nn.CrossEntropyLoss(),
            "position": nn.CrossEntropyLoss(),
            "return": nn.MSELoss(),
            "candle": nn.MSELoss(),
        }
        total_loss = 0
        loss_info = {}
        for key in outputs:
            if key not in y_dict:
                continue
            pred = outputs[key]
            target = y_dict[key]
            if key in ["direction", "position"]:
                # CrossEntropyLoss expects (batch, num_classes), target (batch,)
                loss = loss_fns[key](pred, target.long())
            else:
                # MSELoss: shape (batch, ...)
                loss = loss_fns[key](pred, target)
            weight = 1.0 if loss_weights is None or key not in loss_weights else loss_weights[key]
            total_loss += weight * loss
            loss_info[key] = loss.item()
        return total_loss, loss_info
