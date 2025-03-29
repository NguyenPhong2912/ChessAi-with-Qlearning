import os
# Thiết lập biến môi trường để ẩn thông báo lỗi IMKCFRunLoopWakeUpReliable
os.environ['SDL_AUDIODRIVER'] = 'dummy'

from scr.game.ChessBoard import ChessBoard
from scr.agents.q_learning_agent import QLearningAgent
from scr.models.q_network import QNetwork
from scr.visualization.ChessVisualizer import ChessVisualizer

def main():
    # Khởi tạo bàn cờ
    board = ChessBoard()
    
    # Khởi tạo AI
    state_size = 64 * 12  # 64 ô, mỗi ô có 12 trạng thái có thể (6 loại quân * 2 màu)
    action_size = 64 * 64  # Số lượng nước đi có thể (từ ô này đến ô kia)
    q_network = QNetwork(state_size, action_size)
    agent = QLearningAgent(state_size, action_size)
    
    # Khởi tạo và chạy giao diện
    visualizer = ChessVisualizer(board, agent, q_network)
    visualizer.run()

if __name__ == "__main__":
    main()
