
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def main():
    processor = DataProcessor()
    trainer = ModelTrainer()

    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_dir, 'train.csv')
    test_path = os.path.join(base_dir, 'test.csv')

    print(f"Loading data from {train_path}...")
    try:
        train_df = processor.load_data(train_path)
        test_df = processor.load_data(test_path)
    except FileNotFoundError as e:
        print(e)
        return

    print("Preprocessing training data...")
    X_train, y_train = processor.prepare_data(train_df)
    
    print("Preprocessing test data...")
    X_test, y_test = processor.prepare_data(test_df)

    # Train
    trainer.train(X_train, y_train)

    # Evaluate
    trainer.evaluate(X_test, y_test)

    # Save
    trainer.save_model()

if __name__ == "__main__":
    main()
