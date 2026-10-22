"""BiLSTM + attention classifier (paper Sec. III-D).

Module names are kept identical to the original code so the 2024
checkpoints in ``models/`` load without conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FEATURE_DIM


class AttentionLayer(nn.Module):
    """Additive attention pooling over time."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states):
        attention_weights = F.softmax(self.attention(hidden_states), dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended, attention_weights


class SignLanguageModel(nn.Module):
    def __init__(self, input_size=FEATURE_DIM, hidden_size=256, num_layers=2, num_classes=24, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = AttentionLayer(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """x: (batch, frames, features) -> (logits, attention weights)."""
        lstm_out, _ = self.lstm(self.layer_norm(x))
        attended, attention_weights = self.attention(lstm_out)
        return self.classifier(attended), attention_weights


def load_checkpoint(path, model_config: dict | None = None, device="cpu"):
    """Load a checkpoint saved by ``scripts/train.py`` or the original 2024 notebook.

    Returns (model, checkpoint_dict). The class count is read from the weights.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    cfg = dict(model_config or checkpoint.get("model_config") or {})
    cfg["num_classes"] = state["classifier.3.weight"].shape[0]
    model = SignLanguageModel(**cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint
