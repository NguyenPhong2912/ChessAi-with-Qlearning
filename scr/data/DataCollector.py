import os
import json
import numpy as np
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="data/training"):
        """
        Khởi tạo DataCollector để thu thập và lưu trữ dữ liệu huấn luyện.
        
        Args:
            data_dir (str): Thư mục lưu trữ dữ liệu huấn luyện.
        """
        self.data_dir = data_dir
        self.current_game_data = {
            'moves': [],
            'states': [],
            'rewards': [],
            'game_result': None,
            'timestamp': None
        }
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        """Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def add_move(self, state_vector, move, reward):
        """
        Thêm một nước đi vào dữ liệu game hiện tại.
        
        Args:
            state_vector (np.ndarray): Vector trạng thái bàn cờ.
            move (str): Nước đi dạng 'e2e4'.
            reward (float): Phần thưởng nhận được.
        """
        self.current_game_data['states'].append(state_vector.tolist())
        self.current_game_data['moves'].append(move)
        self.current_game_data['rewards'].append(float(reward))
        
    def set_game_result(self, winner):
        """
        Cập nhật kết quả ván đấu.
        
        Args:
            winner (str): Người chiến thắng ('white', 'black', hoặc 'draw').
        """
        self.current_game_data['game_result'] = winner
        
    def save_game_data(self):
        """Lưu dữ liệu ván đấu hiện tại vào file."""
        if not self.current_game_data['moves']:
            return  # Không lưu nếu không có nước đi nào
            
        # Thêm timestamp
        self.current_game_data['timestamp'] = datetime.now().isoformat()
        
        # Tạo tên file với timestamp
        filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Lưu dữ liệu vào file JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_game_data, f, indent=2)
            
        # Reset dữ liệu game
        self.current_game_data = {
            'moves': [],
            'states': [],
            'rewards': [],
            'game_result': None,
            'timestamp': None
        }
        
    @staticmethod
    def load_training_data(data_dir="data/training"):
        """
        Đọc tất cả dữ liệu huấn luyện từ thư mục.
        
        Args:
            data_dir (str): Thư mục chứa dữ liệu huấn luyện.
            
        Returns:
            tuple: (states, moves, rewards, results) - Dữ liệu huấn luyện đã được xử lý.
        """
        all_states = []
        all_moves = []
        all_rewards = []
        all_results = []
        
        if not os.path.exists(data_dir):
            return np.array([]), [], np.array([]), []
            
        for filename in os.listdir(data_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                
            all_states.extend(game_data['states'])
            all_moves.extend(game_data['moves'])
            all_rewards.extend(game_data['rewards'])
            all_results.extend([game_data['game_result']] * len(game_data['moves']))
            
        return (np.array(all_states), all_moves, 
                np.array(all_rewards), all_results)
                
    def get_statistics(self):
        """
        Tính toán thống kê từ dữ liệu đã thu thập.
        
        Returns:
            dict: Các thống kê về dữ liệu huấn luyện.
        """
        all_data = self.load_training_data(self.data_dir)
        states, moves, rewards, results = all_data
        
        if len(states) == 0:
            return {
                'total_games': 0,
                'total_moves': 0,
                'avg_moves_per_game': 0,
                'white_wins': 0,
                'black_wins': 0,
                'draws': 0,
                'avg_reward': 0,
                'max_reward': 0,
                'min_reward': 0
            }
            
        # Tính số ván đấu duy nhất
        unique_timestamps = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    game_data = json.load(f)
                    if game_data['timestamp']:
                        unique_timestamps.add(game_data['timestamp'])
                        
        total_games = len(unique_timestamps)
        
        # Tính thống kê khác
        stats = {
            'total_games': total_games,
            'total_moves': len(moves),
            'avg_moves_per_game': len(moves) / total_games if total_games > 0 else 0,
            'white_wins': results.count('white'),
            'black_wins': results.count('black'),
            'draws': results.count('draw'),
            'avg_reward': float(np.mean(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards))
        }
        
        return stats 