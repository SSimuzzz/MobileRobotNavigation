class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Pre-norm residual block
        h = self.fc1(F.relu(self.ln1(x)))
        h = self.fc2(F.relu(self.ln2(h)))
        return x + h

class DQN(nn.Module):
    def __init__(self, inputs: int, outputs: int, dim: int = 128, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(inputs, dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, outputs)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)