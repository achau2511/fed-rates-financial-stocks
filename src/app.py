"""
Purpose:
Interactive web dashboard for exploring historical data
and generating predictions using the trained model.

Built with Streamlit.

Responsibilities:
- Load processed data and trained model
- Visualize rate decisions and stock performance
- Allow users to test hypothetical scenarios
- Display model predictions and insights
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns