"""Data processing module for the student performance predictor."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE

from src.config import DATA_PATH, FEATURES

class DataProcessor:
    """Handles data loading, preprocessing, and dataset splitting.
    
    Attributes:
        data_path (str): Path to the dataset CSV file.
        feature_cols (List[str]): List of model feature column names.
        scaler (RobustScaler): Instance used to scale the data.
        poly (PolynomialFeatures): Instance used for polynomial feature generation.
        feature_names (Optional[np.ndarray]): Names of the newly generated features.
    """
    
    def __init__(self, data_path: str = str(DATA_PATH)) -> None:
        """Initializes the DataProcessor with paths and transformers."""
        self.data_path = data_path
        self.feature_cols = FEATURES
        self.scaler = RobustScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_names: Optional[np.ndarray] = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads dataset from the specified path.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        df = pd.read_csv(self.data_path)
        print(f"Loaded: {df.shape[0]:,} rows | Columns: {df.columns.tolist()}")
        return df
        
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes outliers from specific numeric columns using the IQR method.
        
        Args:
            df (pd.DataFrame): The dataframe to process.
            
        Returns:
            pd.DataFrame: A new dataframe with outliers removed.
        """
        cols_iqr = [
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation",
            "total_score",
        ]
        
        # Only process columns that exist in the dataframe
        cols_to_process = [col for col in cols_iqr if col in df.columns]
        
        for col in cols_to_process:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        print(f"After IQR outlier removal: {df.shape[0]:,} rows")
        return df
        
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
        """Preprocesses the data: creates target variable, clips percentiles, and scales features.
        
        Args:
            df (pd.DataFrame): Raw dataframe.
            
        Returns:
            Tuple containing:
                - X_poly (np.ndarray): Scaled and polynomially transformed features.
                - y (pd.Series): Target variable array.
                - df (pd.DataFrame): The dataframe with the new result column.
        """
        # Grade → pass/fail
        df["result"] = df["grade"].apply(lambda g: 1 if g in ["A", "B", "C"] else 0)
        vc = df["result"].value_counts()
        print(f"Class distribution → Pass: {vc.get(1,0):,} ({vc.get(1,0)/len(df)*100:.1f}%)  "
              f"Fail: {vc.get(0,0):,} ({vc.get(0,0)/len(df)*100:.1f}%)")

        X_raw = df[self.feature_cols].copy()
        y     = df["result"].copy()

        # Clip at 1st/99th percentile
        for col in self.feature_cols:
            X_raw[col] = X_raw[col].clip(X_raw[col].quantile(0.01), X_raw[col].quantile(0.99))

        X_scaled = self.scaler.fit_transform(X_raw)
        X_poly   = self.poly.fit_transform(X_scaled)
        self.feature_names = self.poly.get_feature_names_out(self.feature_cols)
        
        return X_poly, y, df

    def split_and_balance(self, X_poly: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Splits data into train/test sets and applies SMOTE balancing to training set.
        
        Args:
            X_poly (np.ndarray): The feature array.
            y (pd.Series): The target variable.
            
        Returns:
            Tuple containing balanced X_train, X_test, balanced y_train, and y_test.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        vc_sm = pd.Series(y_train_sm).value_counts()
        print(f"After SMOTE → Pass: {vc_sm.get(1,0):,}  Fail: {vc_sm.get(0,0):,}")
        
        return X_train_sm, X_test, y_train_sm, y_test
