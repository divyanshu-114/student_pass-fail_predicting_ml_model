"""Inference module for predicting student performance."""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.config import MODELS_DIR, FEATURES

class StudentPredictor:
    """Handles loading pretrained models and running predictions.
    
    Attributes:
        models_dir (str): Directory containing the trained models.
        model_files (Dict[str, str]): Mapping of model names to file paths.
        bundle (Optional[Dict[str, Any]]): Loaded models and transformers bundle.
        feature_cols (List[str]): List of feature column names.
    """
    
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        """Initializes the predictor with model paths."""
        self.models_dir = models_dir
        self.model_files = {
            "model":  os.path.join(self.models_dir, "model.pkl"),
            "scaler": os.path.join(self.models_dir, "scaler.pkl"),
            "poly":   os.path.join(self.models_dir, "poly.pkl"),
            "kmeans": os.path.join(self.models_dir, "kmeans.pkl"),
            "meta":   os.path.join(self.models_dir, "meta.pkl"),
        }
        self.bundle = self._load_pretrained_model()
        self.feature_cols = FEATURES

    def _load_pretrained_model(self) -> Optional[Dict[str, Any]]:
        """Loads machine learning models from disk.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary of loaded artifacts or None if missing.
        """
        if not all(os.path.exists(v) for v in self.model_files.values()):
            return None
        return {
            "model":  joblib.load(self.model_files["model"]),
            "scaler": joblib.load(self.model_files["scaler"]),
            "poly":   joblib.load(self.model_files["poly"]),
            "kmeans": joblib.load(self.model_files["kmeans"]),
            "meta":   joblib.load(self.model_files["meta"]),
        }

    def predict_bundle(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the data through the transformation and prediction pipeline.
        
        Args:
            X_df (pd.DataFrame): Raw input dataframe containing student data.
            
        Returns:
            Tuple containing probabilities, pass/fail predictions, and cluster assignments.
        """
        if not self.bundle:
            raise RuntimeError("Models are not loaded. Run training first.")
            
        Xc = X_df.copy()
        for col in self.feature_cols:
            Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(0.0)
            
        X_s      = self.bundle["scaler"].transform(Xc[self.feature_cols])
        X_p      = self.bundle["poly"].transform(X_s)
        probas   = self.bundle["model"].predict_proba(X_p)[:, 1]
        preds    = (probas >= 0.5).astype(int)
        clusters = self.bundle["kmeans"].predict(X_s)
        return probas, preds, clusters

    def get_student_recommendations(self, row: pd.Series) -> List[str]:
        """Provides actionable recommendations based on student metrics.
        
        Args:
            row (pd.Series): A single row of student data.
            
        Returns:
            List[str]: List of recommendations.
        """
        recs = []
        if float(row.get("weekly_self_study_hours", 20)) < 15:
            recs.append("📚 Increase weekly self-study to **15–20 hours** with a consistent schedule.")
        if float(row.get("attendance_percentage", 100)) < 80:
            recs.append("🏫 Aim for **80%+ attendance** to reduce missed concepts.")
        if float(row.get("class_participation", 10)) < 5:
            recs.append("🙋 Improve participation: ask questions + attempt in-class problems.")
        if not recs:
            recs.append("🌟 Keep it up! Maintain consistency and add spaced revision weekly.")
        return recs

    def get_meta(self) -> Dict[str, Any]:
        """Returns metadata associated with the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary of metadata.
        """
        return self.bundle["meta"] if self.bundle else {}

    def is_ready(self) -> bool:
        """Checks if all models are successfully loaded.
        
        Returns:
            bool: True if models are ready for inference, otherwise False.
        """
        return self.bundle is not None
