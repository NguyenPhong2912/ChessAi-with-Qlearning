import torch
import random
import numpy as np

import torch.optim as optim

class Trainer:
    def __init__(self, agent, episodes: int, learning_rate: float = 0.001):
        """
        Huấn luyện tác nhân Q-learning trong một số lượng tập nhất định.

        Args:
            agent: Đối tượng QLearningAgent cần được huấn luyện.
            episodes (int): Số lượng tập huấn luyện.
            learning_rate (float): Tốc độ học của mô hình.
        """
        self.agent = agent
        self.episodes = episodes
        self.optimizer = optim.Adam(agent.q_network.parameters(), lr=learning_rate)

    def train(self):
        """
        Chạy quá trình huấn luyện bằng cách cho tác nhân chơi nhiều tập.
        """
        for episode in range(self.episodes):
            state = self.agent.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.agent.env.step(action)
                self.agent.update_q_values(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}/{self.episodes} - Total Reward: {total_reward}")

    def save_progress(self, filename: str):
        """
        Lưu trạng thái mô hình hiện tại.

        Args:
            filename (str): Đường dẫn file để lưu mô hình.
        """
        torch.save(self.agent.q_network.state_dict(), filename)
        print(f"Model saved to {filename}")
