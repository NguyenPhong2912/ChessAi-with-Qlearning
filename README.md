# Chess AI with Q-Learning

Dự án này là một AI cờ vua đơn giản sử dụng Q-Learning. Hiện tại, AI có thể chơi thử nghiệm với các nước đi ngẫu nhiên.

## 📁 Cấu trúc dự án

chess_ai_project/
│── src/
│   │── agents/
│   │   │── q_learning_agent.py
│   │   │── experience_replay.py
│   │── models/
│   │   │── q_network.py
│   │── utils/
│   │   │── move_encoder.py
│   │   │── callback_logger.py
│   │── training/
│   │   │── trainer.py
│   │   │── model_evaluator.py
│   │── game/
│   │   │── chess_board.py
│   │   │── chess_engine.py
│   │── visualization/
│   │   │── chess_visualizer.py
│── data/
│   │── saved_models/
│   │── logs/
│── notebooks/
│── tests/
│── main.py
│── requirements.txt
│── README.md


## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt

2. Chạy chương trình
- python main.py

🛠️ Chức năng chính

game/: Xử lý logic cờ vua.
agents/: AI học Q-Learning (đang phát triển).
models/: Mô hình mạng nơ-ron cho AI.
training/: Huấn luyện AI.
visualization/: Hiển thị bàn cờ.
🎯 TODO

 Cập nhật luật di chuyển của quân cờ.
 Thêm Q-Learning để AI có thể học chiến lược.
 Cải thiện hiển thị trực quan.


---

## 📝 **Ghi chú**
- **`main.py`**: Chạy chương trình và hiển thị bàn cờ.
- **`requirements.txt`**: Danh sách thư viện cần cài đặt.
- **`README.md`**: Hướng dẫn sử dụng dự án.

🔥 **Nếu cần bổ sung gì thêm, báo mình nhé!, qua email !ntphong1231@gmail.com** ♟️


