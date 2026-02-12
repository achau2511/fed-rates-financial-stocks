# 📈 Fed Interest Rate Changes & Financial Sector Stock Performance

## 👥 Team Members
Isaac Toffel (Project Lead)  
Alan Chau
Julian Antropow de la Hoz  
Armaan Gupta  
Lucas Azout  
Mahishi Murarka  

---

## 📌 Project Overview

This project examines the relationship between Federal Reserve interest rate decisions and the stock performance of major financial institutions.

By combining macroeconomic policy data from the Federal Reserve with equity market data, we develop machine learning models to determine whether financial sector stocks outperform or underperform the S&P 500 following Federal Open Market Committee (FOMC) rate changes.

The project answers a practical, investor-focused question:

**Can we predict financial sector performance based on shifts in Federal Reserve monetary policy?**

---

## ❓ Research Question

Do Federal Reserve interest rate changes predict short-term stock performance for financial sector companies, and does this relationship vary based on the magnitude and direction of the rate change?

---

## 🎯 Key Objectives

- Build an automated data pipeline to collect Fed rate decisions and daily stock prices  
- Engineer features such as rate change magnitude, volatility windows, and technical indicators  
- Develop classification models predicting relative performance in 30-day windows after rate changes  
- Evaluate model performance using accuracy, precision, recall, and F1-score  
- Create an interactive web application for exploring historical trends and testing hypothetical scenarios  

---

## 📊 Data Sources

### Federal Reserve Economic Data (FRED API)
- Historical Federal Funds Rate  
- FOMC meeting dates  
- Macroeconomic indicators  

Free access with API key.

### Alpha Vantage API
- Daily stock prices for 10–15 major financial institutions:
  - JPMorgan Chase (JPM)
  - Goldman Sachs (GS)
  - Bank of America (BAC)
  - Wells Fargo (WFC)
  - Citigroup (C)
  - Morgan Stanley (MS)
- S&P 500 index data  

Free tier provides 500 API calls per day.

---

## 🏗 Required Deliverables

### Structured Database
PostgreSQL or MongoDB storing:
- FRED data  
- Stock price data  
- Engineered features  

### Machine Learning Model
Trained classification model evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

### Data Pipeline
Automated API collection scripts with:
- Error handling  
- Rate limit management  
- Scheduling capability  

### Web Application
Interactive dashboard (Streamlit or Flask) that displays:
- Historical analysis  
- Visualizations  
- Model predictions  
- Scenario testing  

### GitHub Repository
Complete, well-documented codebase with setup instructions and version control.

---

## 🧠 Modeling Approach

We treat this as a binary classification problem.

Target variable:
1 → Financial sector outperforms the S&P 500 within 30 days of a rate decision  
0 → Underperforms  

Example models:
- Logistic Regression  
- Random Forest  
- Other classification algorithms  

Feature importance analysis is used to understand which macroeconomic and technical indicators most influence predictions.

---

## 🗂 Project Structure

fed-rates-financial-stocks/

├── README.md  
├── requirements.txt  
├── .env.example  
├── .gitignore  
│  
├── src/  
│   ├── data_pipeline.py   # API data collection  
│   ├── features.py        # Feature engineering  
│   ├── model.py           # Model training & evaluation  
│   └── app.py             # Streamlit/Flask dashboard  
│  
├── notebooks/             # Exploratory analysis  
├── data/                  # Raw and processed data (ignored)  
└── tests/                 # Optional unit tests  

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

git clone <repository-url>  
cd fed-rates-financial-stocks  

---

### 2. Create a Virtual Environment

Mac / Linux:

python -m venv venv  
source venv/bin/activate  

Windows:

python -m venv venv  
venv\Scripts\activate  

---

### 3. Install Dependencies

pip install -r requirements.txt  

---

### 4. Add API Keys

Copy the environment template:

cp .env.example .env  

Then add your keys:

FRED_API_KEY=your_key  
ALPHAVANTAGE_API_KEY=your_key  

---

## 🚀 Running the Project

Run data collection:

python src/data_pipeline.py  

Run feature engineering:

python src/features.py  

Train the model:

python src/model.py  

Launch the dashboard:

streamlit run src/app.py  

---

## 📈 Example Features

- Rate change magnitude (basis points)  
- Direction (hike, cut, hold)  
- Rolling returns  
- Volatility windows  
- Technical indicators  
- Relative performance vs S&P 500  

---

## 🌐 Web Application

The dashboard allows users to:

- Explore historical Fed rate changes  
- Visualize stock performance trends  
- View model predictions  
- Test hypothetical rate scenarios  

---

## 🛠 Tech Stack

- Python  
- pandas  
- numpy  
- scikit-learn  
- requests  
- Streamlit or Flask  
- PostgreSQL or MongoDB  

---

## 📌 Expected Outcomes

By the end of the semester, we will:

- Validate whether Fed rate changes provide predictive power  
- Identify the most important macroeconomic and technical features  
- Deploy a live interactive dashboard  
- Deliver a reproducible end-to-end financial data science pipeline  

---

## 📜 License

MIT License