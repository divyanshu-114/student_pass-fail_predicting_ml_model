"""Configuration file for the modeling pipeline and app.

This module defines central configuration paths and feature lists used
across the entirety of the project.
"""

import os
from pathlib import Path

# Identify the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Define input/output paths
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
MODELS_DIR = BASE_DIR / "models"

# Define the features to be used in the model
FEATURES = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
]
