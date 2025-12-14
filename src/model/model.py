import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


class Model(nn.Module):
    def __init__(self, is_bidirectional=True, input_size=768, hidden_size=512, drop_rate=0.3):
        super(Model, self).__init__()

        self.is_bidirectional = is_bidirectional
        num_directions = 2 if is_bidirectional else 1

        # 1. GRU Слой
        self.GRU = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=is_bidirectional,
            num_layers=2,
            batch_first=True
        )

        output_dim = hidden_size * num_directions
        self.binary_exit = nn.Linear(output_dim, 1)

        self.dropout = nn.Dropout(drop_rate)
        self.exit = nn.Sigmoid()

    def forward(self, x):
        _, hidden = self.GRU(x)

        if self.is_bidirectional:
            pooled = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            pooled = hidden[-1, :, :]

        pooled = self.dropout(pooled)

        logits = self.binary_exit(pooled)

        return self.exit(logits)

