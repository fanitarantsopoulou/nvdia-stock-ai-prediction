# NVDA Quant Dashboard

A real-time quantitative analysis and prediction dashboard for NVIDIA ($NVDA) stock. It uses a PyTorch LSTM model for short-term price forecasting and FinBERT for sentiment analysis, served via a FastAPI backend to a reactive Vue.js frontend.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, PyTorch, yfinance, pandas
* **AI/ML:** LSTM Neural Network, FinBERT (Sentiment Analysis)
* **Frontend:** Vue.js 3, Vite, Chart.js, TailwindCSS (Cyberpunk UI)

## ✨ Key Features
* **Real-Time Data Pipeline:** Fetches live market data and news sentiment.
* **AI Inference:** Predicts the next market-open price movement.
* **Timezone & Market State Handling:** Automatically calculates NASDAQ market hours, pre-market, after-hours, and weekends.
* **Reactive Cyberpunk UI:** Dynamic chart rendering with custom visual states.

## 🚀 How to Run Locally

### 1. Backend (FastAPI)
Navigate to the backend directory, install requirements, and start the server:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Frontend (Vue.js)
Navigate to the frontend directory, install dependencies, and start the dev server:
```bash
npm install
npm run dev
```

### 3. Open your browser
Open your browser and navigate to `http://localhost:5173` to view the dashboard.