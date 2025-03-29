import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class AdvancedQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Advanced Q-Network with convolutional layers and residual connections.
        
        Args:
            state_size (int): Size of the state vector (64 * 12 for chess)
            action_size (int): Number of possible actions (64 * 64 for chess)
        """
        super(AdvancedQNetwork, self).__init__()
        
        # Reshape input to 8x8x12 (chess board with piece channels)
        self.reshape = lambda x: x.view(-1, 12, 8, 8)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res3 = ResidualBlock(64, 64)
        
        # Policy head (for move selection)
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, action_size)
        
        # Value head (for position evaluation)
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (policy_output, value_output)
        """
        # Reshape input
        x = self.reshape(state)
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 64)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output between -1 and 1
        
        return policy, value
    
    def get_action(self, state: torch.Tensor, legal_moves: list[int], epsilon: float = 0.0) -> int:
        """
        Get the best action based on the current state.
        
        Args:
            state (torch.Tensor): Current state
            legal_moves (list[int]): List of legal move indices
            epsilon (float): Exploration rate
            
        Returns:
            int: Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.choice(legal_moves)
        
        with torch.no_grad():
            policy, value = self(state)
            
            # Mask illegal moves
            mask = torch.zeros_like(policy)
            mask[0, legal_moves] = 1
            policy = policy * mask
            
            # Get best action
            if policy.sum() > 0:
                return legal_moves[torch.argmax(policy[0, legal_moves]).item()]
            else:
                return np.random.choice(legal_moves)
    
    def evaluate_position(self, state: torch.Tensor) -> float:
        """
        Evaluate the current position.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            float: Position evaluation (-1 to 1)
        """
        with torch.no_grad():
            _, value = self(state)
            return value.item()
    
    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))
        self.eval()

# Example usage
if __name__ == "__main__":
    # Create a sample state tensor (batch_size=1, state_size=64*12)
    state_size = 64 * 12
    action_size = 64 * 64
    state = torch.randn(1, state_size)
    
    # Create and test the network
    network = AdvancedQNetwork(state_size, action_size)
    policy, value = network(state)
    
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")
    
    # Test action selection
    legal_moves = list(range(10))  # Example legal moves
    action = network.get_action(state, legal_moves, epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test position evaluation
    eval = network.evaluate_position(state)
    print(f"Position evaluation: {eval}") 