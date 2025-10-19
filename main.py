# main.py
from src.pipeline_and_model import main as pipeline_and_model
from src.explore_data import main as explore_data

if __name__ == "__main__":
    print("=== Starting data pipeline ===")
    print("=== Starting model training ===")
    print("=== Starting model evaluation ===")
    pipeline_and_model()
    
    print("\n=== Starting data exploration ===")
    explore_data()
    
    print("\nAll tasks completed. Check the 'figures/' folder for outputs.")
