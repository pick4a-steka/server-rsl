import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_layers=2, num_classes=3, dropout=0.0):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: [batch, 12, 132]
        output, h_n = self.gru(x)

        # h_n: [1, batch, 128], т.е [количество слоев, количество батчей, размер слоя]
        last_hidden = h_n[-1]

        logits = self.classifier(last_hidden) # [batch, количество классов]
        return logits