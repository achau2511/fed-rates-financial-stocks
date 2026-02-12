"""
Purpose:
Train and evaluate machine learning models that predict
financial sector performance after Fed rate changes.

Responsibilities:
- Load engineered features
- Split data into train/test sets
- Train classification models (Logistic Regression, Random Forest, etc.)
- Evaluate using accuracy, precision, recall, and F1
- Save trained model to disk for later use by the web app

Output:
model.pkl
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score