import torch
import torch.nn as nn
import torch.nn.functional as F

class QChessNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers for feature extraction (deepened network)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Batch Normalization after each conv layer
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Value head
        self.value_conv = nn.Conv2d(512, 256, kernel_size=1)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers for value prediction
        self.value_fc1 = nn.Linear(256, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Common network: Convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Value head: Apply 1x1 convolution to reduce channel depth
        value = F.relu(self.value_conv(x))

        # Global Average Pooling (to reduce spatial dimensions)
        value = self.global_avg_pool(value)
        
        # Flatten the output for fully connected layers
        value = value.view(value.size(0), -1)

        # Pass through fully connected layers with dropout
        value = F.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = F.relu(self.value_fc2(value))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc3(value))  # Output is between -1 and 1

        return value
