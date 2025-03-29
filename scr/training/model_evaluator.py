import torch
import numpy as np

class ModelEvaluator:
    def __init__(self, agent, test_games: int):
        """
        Đánh giá hiệu suất của tác nhân sau khi huấn luyện.

        Args:
            agent: Đối tượng QLearningAgent cần đánh giá.
            test_games (int): Số lượng ván chơi để kiểm tra.
        """
        self.agent = agent
        self.test_games = test_games
        self.total_rewards = []

    def evaluate(self):
        """
        Chạy quá trình kiểm tra để đánh giá hiệu suất của mô hình.
        """
        self.total_rewards = []

        for game in range(self.test_games):
            state = self.agent.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.agent.choose_action(state, exploit=True)  # Chọn hành động tốt nhất
                state, reward, done, _ = self.agent.env.step(action)
                total_reward += reward

            self.total_rewards.append(total_reward)
            print(f"Game {game + 1}/{self.test_games} - Total Reward: {total_reward}")

        avg_reward = np.mean(self.total_rewards)
        print(f"Average Reward over {self.test_games} games: {avg_reward}")

    def generate_report(self, filename: str):
        """
        Xuất kết quả đánh giá vào file.

        Args:
            filename (str): Tên file để lưu báo cáo.
        """
        with open(filename, 'w') as f:
            f.write(f"Evaluation Results over {self.test_games} games:\n")
            f.write(f"Average Reward: {np.mean(self.total_rewards)}\n")
        print(f"Report saved to {filename}")
