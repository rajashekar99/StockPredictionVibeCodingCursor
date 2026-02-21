# India Stock Market AI Prediction Tool Using vibe coding

This is a Streamlit-based web app that fetches **Indian stock market (NSE/BSE)** historical data, pulls recent news, and uses **OpenAI** to generate **sentiment analysis** and a **short-term outlook** for selected stocks.

> **Disclaimer:** This project is for educational and experimental purposes only and **is not financial advice**.

## Features

- **Historical data (NSE/BSE)** via `yfinance`
- **News feed** via RSS (Economic Times, Livemint) — no API key required
- **News sentiment analysis** using OpenAI
- **AI-generated short-term outlook** for a stock
- **Chatbot** to ask questions about the selected stock using the same context

## Setup

### 1. Create and activate a virtual environment

```bash
cd c:\Users\HP\cursorVibeCoding
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env
```

Edit `.env`:

- `OPENAI_API_KEY`: your OpenAI API key (required for AI features)

News comes from free RSS feeds (Economic Times, Livemint); no API key needed. Alternatively, set variables directly in your shell environment.

## Running the app

From the project root:

```bash
streamlit run app.py
```

Then open the URL that Streamlit prints (typically `http://localhost:8501`).

## Usage

- Use the **sidebar** to select a popular NSE stock and a date range.
- View the **historical price chart** and key metrics.
- Check **News & Sentiment** to see recent headlines and their aggregated sentiment.
- Open **AI Outlook** to get an LLM-generated short-term view (requires OpenAI key).
- Use the **Chatbot** tab to ask stock-related questions based on the same context.

## Notes

- Historical prices come from Yahoo Finance via `yfinance`; there may be delays or inaccuracies.
- News is sourced from RSS feeds (Economic Times, Livemint); the app is designed as a demo, not for production trading.

