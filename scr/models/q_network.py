import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 64):
        """
        QNetwork được sử dụng để xấp xỉ Q-values cho mỗi trạng thái và hành động.
        
        Args:
            state_size (int): Kích thước vector trạng thái (số lượng đặc trưng của trạng thái).
            action_size (int): Số lượng hành động khả dụng.
            hidden_dim (int): Số lượng nơ-ron trong mỗi lớp ẩn. Mặc định là 64.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Truyền xuôi của mạng, nhận đầu vào là trạng thái và trả về Q-values cho mỗi hành động.
        
        Args:
            state (torch.Tensor): Vector trạng thái với shape (batch_size, state_size).
        
        Returns:
            torch.Tensor: Q-values cho mỗi hành động với shape (batch_size, action_size).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
