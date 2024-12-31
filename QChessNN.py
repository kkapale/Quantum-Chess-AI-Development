import torch
import torch.nn as nn
import torch.nn.functional as F

class QChessNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)  # 12 channels for the board state
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        """
        # Policy head
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.policy_fc = nn.Linear(64 * 8 * 8, 4672)  # 4672 possible moves in chess (move encoding)
        """
        # Value head
        self.value_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.value_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Common network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return value