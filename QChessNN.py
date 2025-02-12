import torch
import torch.nn as nn
import torch.nn.functional as F

class QChessNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature Extraction
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Added deeper layers
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  

        # Batch Normalization for stability
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)

        # Value Head
        self.value_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.value_fc1 = nn.Linear(128 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Policy Head
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)  
        self.policy_fc1 = nn.Linear(128 * 8 * 8, 256)  
        self.policy_fc2 = nn.Linear(256, 128)  

        # Separate outputs for move components
        self.piece_head = nn.Linear(128, 1)  
        self.pos1_head = nn.Linear(128, 1)   
        self.pos2_head = nn.Linear(128, 1)
        self.pos3_head = nn.Linear(128, 1)
        self.move_type_head = nn.Linear(128, 1)  
        self.variation_head = nn.Linear(128, 1)  
        self.promotion_head = nn.Linear(128, 1)  

    def forward(self, x):
        # Feature Extraction with Batch Normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Value Head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Outputs in range [-1, 1]
        
        # Policy Head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = F.relu(self.policy_fc1(policy))
        policy = F.relu(self.policy_fc2(policy))

        # Predict discrete action components
        piece = F.softmax(self.piece_head(policy), dim=1)
        pos1 = F.softmax(self.pos1_head(policy), dim=1)
        pos2 = F.softmax(self.pos2_head(policy), dim=1)
        pos3 = F.softmax(self.pos3_head(policy), dim=1)
        move_type = F.softmax(self.move_type_head(policy), dim=1)
        variation = F.softmax(self.variation_head(policy), dim=1)
        promotion_piece = F.softmax(self.promotion_head(policy), dim=1)
        return value, (piece, pos1, pos2, pos3, move_type, variation, promotion_piece)
        
