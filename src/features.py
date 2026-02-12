"""
Purpose:
Create model-ready features from raw economic and stock data.

This script transforms raw price and rate data into signals that
machine learning models can use.

Responsibilities:
- Calculate rate change magnitude and direction
- Compute rolling returns and volatility windows
- Generate technical indicators
- Build target labels (outperform vs underperform)
- Output a final training dataset
"""

import pandas as pd
import numpy as np