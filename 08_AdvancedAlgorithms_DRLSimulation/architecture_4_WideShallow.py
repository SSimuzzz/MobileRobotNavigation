import torch.nn as nn

class WideShallowDQN(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, width),
            nn.ReLU(),
            nn.Dropout(dropout),          # opzionale (puoi lasciare 0.0)
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, outputs)
        )

    def forward(self, x):
        return self.net(x)
