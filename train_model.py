"""Script to train and save the student performance models."""

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def main() -> None:
    print("🚀 Starting training pipeline...")
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = processor.load_data()
    
    # 2. Remove Outliers
    print("\n[2] Removing outliers...")
    df_clean = processor.remove_outliers(df)
    
    # 3. Preprocess and Extract Features
    print("\n[3] Preprocessing features and target...")
    X_poly, y, df_clean = processor.preprocess(df_clean)
    
    # 4. Split and Balance Data
    print("\n[4] Splitting and balancing data with SMOTE...")
    X_train_sm, X_test, y_train_sm, y_test = processor.split_and_balance(X_poly, y)
    
    # 5. Train and Evaluate Models
    print("\n[5] Training and evaluating candidate models...")
    trainer.train_and_evaluate(X_train_sm, y_train_sm, X_test, y_test)
    
    # 6. Train Clustering for Segmentation
    print("\n[6] Training KMeans clustering...")
    trainer.train_kmeans(df_clean, processor.feature_cols, processor.scaler)
    
    # 7. Save All Artifacts
    print("\n[7] Saving artifacts...")
    trainer.save_artifacts(processor.scaler, processor.poly, processor.feature_names, processor.feature_cols)

if __name__ == "__main__":
    main()