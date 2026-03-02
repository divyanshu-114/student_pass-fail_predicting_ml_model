"""Model training module for the student performance predictor."""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from src.config import MODELS_DIR

class ModelTrainer:
    """Trains and evaluates machine learning models.
    
    Attributes:
        models_dir (str): Directory where models and artifacts will be saved.
        best_model (Any): The best estimator identified post-training.
        best_name (str): The name of the best performing model.
        kmeans (Optional[KMeans]): The fitted KMeans clustering model.
        metrics (Dict[str, float]): Dictionary containing performance metrics of the best model.
    """
    
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        """Initializes ModelTrainer with the output directory."""
        self.models_dir = models_dir
        self.best_model: Any = None
        self.best_name: str = ""
        self.kmeans: Optional[KMeans] = None
        self.metrics: Dict[str, float] = {}
        
    def train_and_evaluate(self, X_train_sm: np.ndarray, y_train_sm: pd.Series, 
                           X_test: np.ndarray, y_test: pd.Series) -> None:
        """Trains multiple candidate models using GridSearchCV and evaluates them.
        
        Identifies and saves the best model based on weighted F1 score.
        
        Args:
            X_train_sm (np.ndarray): Balanced training feature array.
            y_train_sm (pd.Series): Balanced training target series.
            X_test (np.ndarray): Testing feature array.
            y_test (pd.Series): Testing target series.
        """
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        candidates = {
            "LogisticRegression": {
                "model": LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000),
                "params": {"C": [0.1, 1.0, 10.0]}
            },
            "RandomForest": {
                "model": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
                "params": {"n_estimators": [50, 100], "max_depth": [6, 12]}
            },
            "XGBoost": {
                "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
                "params": {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
            }
        }
        
        print("\n{:<20} {:>10} {:>12} {:>10} {:>12}".format("Model", "Accuracy", "Precision", "Recall", "F1 (weighted)"))
        print("-" * 66)

        best_f1 = -1.0
        best_params: Dict[str, Any] = {}

        for name, config in candidates.items():
            print(f"\nRunning GridSearchCV for {name}...")
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring="f1_weighted",
                cv=cv_strategy,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_sm, y_train_sm)
            best_estimator = grid_search.best_estimator_
            print(f"Best params for {name}: {grid_search.best_params_}")
            
            y_pred = best_estimator.predict(X_test)
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec  = recall_score(y_test, y_pred, average="weighted")
            f1   = f1_score(y_test, y_pred, average="weighted")

            print("{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>12.4f}".format(name, acc, prec, rec, f1))

            if f1 > best_f1:
                best_f1, self.best_name, self.best_model, best_params = f1, name, best_estimator, grid_search.best_params_

        print(f"\n🏆 Best overall model: {self.best_name}  (F1 = {best_f1:.4f})")
        print(f"Winning hyperparameters: {best_params}")

        # Final evaluation on test set
        y_pred_final = self.best_model.predict(X_test)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_final)),
            "precision": float(precision_score(y_test, y_pred_final, average="weighted")),
            "recall": float(recall_score(y_test, y_pred_final, average="weighted")),
            "f1": float(f1_score(y_test, y_pred_final, average="weighted"))
        }

        print("\n── Final Metrics (" + self.best_name + ") ──────────────────")
        print(f"  Accuracy  : {self.metrics['accuracy']:.4f}")
        print(f"  Precision : {self.metrics['precision']:.4f}")
        print(f"  Recall    : {self.metrics['recall']:.4f}")
        print(f"  F1 Score  : {self.metrics['f1']:.4f}")

        print("\n── Classification Report ──────────────────────────────")
        print(classification_report(y_test, y_pred_final, target_names=["Fail", "Pass"]))

        cm = confusion_matrix(y_test, y_pred_final, labels=[0, 1])
        print("── Confusion Matrix ───────────────────────────────────")
        print(f"  TN (correct fail) : {cm[0,0]:>8,}")
        print(f"  FP (fail→pass)    : {cm[0,1]:>8,}")
        print(f"  FN (pass→fail)    : {cm[1,0]:>8,}")
        print(f"  TP (correct pass) : {cm[1,1]:>8,}")

    def train_kmeans(self, df: pd.DataFrame, feature_cols: List[str], scaler: Any) -> None:
        """Trains a KMeans clustering model for student segmentation.
        
        Args:
            df (pd.DataFrame): Training dataframe.
            feature_cols (List[str]): Features to utilize for clustering.
            scaler (Any): Pre-fitted scaler instance.
        """
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        self.kmeans.fit(scaler.transform(df[feature_cols]))

    def save_artifacts(self, scaler: Any, poly: Any, feature_names: np.ndarray, feature_cols: List[str]) -> None:
        """Saves the best model, transformers, and metrics to disk.
        
        Args:
            scaler (Any): Pre-fitted model scaler.
            poly (Any): Pre-fitted polynomial features transformer.
            feature_names (np.ndarray): Extracted feature names.
            feature_cols (List[str]): Original feature column names.
        """
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(self.models_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(poly, os.path.join(self.models_dir, "poly.pkl"))
        joblib.dump(self.kmeans, os.path.join(self.models_dir, "kmeans.pkl"))
        joblib.dump(feature_names.tolist(), os.path.join(self.models_dir, "feature_names.pkl"))
        joblib.dump(feature_cols, os.path.join(self.models_dir, "input_feature_cols.pkl"))
        joblib.dump({
            "best_model_name": self.best_name,
            **self.metrics
        }, os.path.join(self.models_dir, "meta.pkl"))
        print("\n✅ All artifacts saved to models/")
