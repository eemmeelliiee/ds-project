# main.py
from pipeline_water_quality import main as train_model
from explore_data import main as explore_data

if __name__ == "__main__":
    print("=== Starting data pipeline ===")
    train_model()
    
    print("\n=== Starting data exploration ===")
    explore_data()
    
    print("\nAll tasks completed. Check the 'figures/' folder for outputs.")
