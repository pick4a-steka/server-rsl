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

        self.attention = nn.Linear(hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        output, h_n = self.gru(x)
        # output: [batch, frames, hidden_size]

        scores = self.attention(output)
        # scores: [batch, frames, 1]

        weights = torch.softmax(scores, dim=1)
        # weights: [batch, frames, 1]

        context = (output * weights).sum(dim=1)
        # context: [batch, hidden_size]

        logits = self.classifier(context)
        return logits