from scr.models.ModelTrainer import ModelTrainer

def main():
    # Khởi tạo trainer
    trainer = ModelTrainer(
        state_size=768,  # 64 ô * 12 loại quân
        action_size=4096,  # 64 ô nguồn * 64 ô đích
        hidden_dim=256,
        learning_rate=0.001,
        batch_size=32
    )
    
    # Huấn luyện mô hình
    trainer.train(
        num_epochs=100,
        data_dir="data/training",
        save_path="models/q_network.pth"
    )

if __name__ == "__main__":
    main() 