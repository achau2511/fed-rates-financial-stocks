"""
Purpose:
Collect and store all raw data used in the project.

This script connects to:
- FRED API for Federal Funds Rate + FOMC meeting data
- Alpha Vantage API for daily stock prices

Responsibilities:
- Fetch macroeconomic and stock data
- Handle API rate limits and errors
- Clean and standardize formats
- Save results to CSV files or the project database
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

def fetch_fred():
    pass

def fetch_prices():
    pass

def save_data():
    pass