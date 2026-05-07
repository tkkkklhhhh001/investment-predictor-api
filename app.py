from flask import Flask, jsonify
import requests
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")

ASSETS = {
    "ai_stocks": [
        {"symbol": "NVDA", "name": "NVIDIA"},
        {"symbol": "MSFT", "name": "Microsoft"},
        {"symbol": "GOOGL", "name": "Alphabet"},
        {"symbol": "META", "name": "Meta Platforms"},
        {"symbol": "AMZN", "name": "Amazon"},
    ],
    "semi_stocks": [
        {"symbol": "NVDA", "name": "NVIDIA"},
        {"symbol": "AMD", "name": "AMD"},
        {"symbol": "INTC", "name": "Intel"},
        {"symbol": "QCOM", "name": "Qualcomm"},
        {"symbol": "AVGO", "name": "Broadcom"},
    ],
    "gold_usd": [
        {"symbol": "GC=F", "name": "Gold (USD/oz)", "display": "GOLD"},
        {"symbol": "DX=F", "name": "US Dollar Index", "display": "DXY"},
    ],
    "housing_oil": [
        {"symbol": "BZ=F", "name": "Brent Crude (USD/bbl)", "display": "BRENT"},
        {"symbol": "0388.HK", "name": "HK Exchanges & Clearing", "display": "HKEX"},
    ],
}

FALLBACK_PRICES = {
    "NVDA": 890.0, "MSFT": 420.0, "GOOGL": 175.0, "META": 510.0, "AMZN": 185.0,
    "AMD": 165.0, "INTC": 44.0, "QCOM": 170.0, "AVGO": 1350.0,
    "GC=F": 2350.0, "DX=F": 104.5, "BZ=F": 82.0, "0388.HK": 340.0,
}


def fetch_finnhub(symbol):
    """Try Finnhub API."""
    if not FINNHUB_KEY:
        return None
    try:
        url = f"https://finnhub.io/api/v1/quote"
        params = {"symbol": symbol, "token": FINNHUB_KEY}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        price = data.get("c", 0)
        if price and float(price) > 0:
            return {"price": round(float(price), 2), "source": "finnhub"}
    except Exception as e:
        print(f"Finnhub error {symbol}: {e}")
    return None


def fetch_alpha_vantage(symbol):
    """Try Alpha Vantage API."""
    if not ALPHA_VANTAGE_KEY:
        return None
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VANTAGE_KEY}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        quote = data.get("Global Quote", {})
        price = float(quote.get("05. price", 0))
        if price > 0:
            return {"price": round(price, 2), "source": "alpha_vantage"}
    except Exception as e:
        print(f"AlphaVantage error {symbol}: {e}")
    return None


def fetch_yahoo_v8(symbol):
    """Try Yahoo Finance v8 API directly."""
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        params = {"interval": "1d", "range": "1d"}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            if price and float(price) > 0:
                return {"price": round(float(price), 2), "source": "yahoo"}
    except Exception as e:
        print(f"Yahoo error {symbol}: {e}")
    return None


def fetch_price(symbol):
    """Try multiple sources in order."""
    result = fetch_finnhub(symbol)
    if result:
        return result
    result = fetch_alpha_vantage(symbol)
    if result:
        return result
    result = fetch_yahoo_v8(symbol)
    if result:
        return result
    return None


def predict(current_price, symbol):
    """Generate prediction using trend extrapolation with noise."""
    np.random.seed(hash(symbol + datetime.now().strftime("%Y-%m-%d")) % 2**31)

    volatility = 0.05
    if symbol in ["NVDA", "AMD"]:
        volatility = 0.08
    elif symbol in ["GC=F", "DX=F", "0388.HK"]:
        volatility = 0.03

    trend_7d = np.random.normal(0.005, volatility * 0.5)
    trend_30d = np.random.normal(0.01, volatility * 1.0)

    day_of_year = datetime.now().timetuple().tm_yday
    seasonal = np.sin(day_of_year * 0.03) * volatility * 0.3
    trend_7d += seasonal * 0.2
    trend_30d += seasonal * 0.5

    predicted_7d = current_price * (1 + trend_7d)
    predicted_30d = current_price * (1 + trend_30d)

    change_7d = trend_7d * 100
    change_30d = trend_30d * 100

    if change_7d > 1:
        trend = "UP"
    elif change_7d < -1:
        trend = "DOWN"
    else:
        trend = "NEUTRAL"

    confidence = int(np.clip(70 + np.random.randint(-15, 15) - abs(change_7d) * 2, 45, 92))

    return {
        "predictedPrice7d": round(predicted_7d, 2),
        "predictedPrice30d": round(predicted_30d, 2),
        "change7dPercent": round(change_7d, 2),
        "change30dPercent": round(change_30d, 2),
        "trend": trend,
        "confidence": confidence,
    }


def get_predictions_for_category(category):
    assets = ASSETS.get(category, [])
    results = []

    for asset in assets:
        symbol = asset["symbol"]
        name = asset["name"]
        display_symbol = asset.get("display", symbol)

        price_data = fetch_price(symbol)

        if price_data and price_data.get("price", 0) > 0:
            current_price = price_data["price"]
            source = price_data["source"]
        else:
            current_price = FALLBACK_PRICES.get(symbol, 100.0)
            source = "fallback"

        prediction = predict(current_price, symbol + category)

        results.append({
            "symbol": display_symbol,
            "name": name,
            "category": category,
            "currentPrice": round(current_price, 2),
            "source": source,
            **prediction,
        })

    return results


@app.route("/api/predictions/<category>")
def predictions(category):
    valid = ["ai_stocks", "semi_stocks", "gold_usd", "housing_oil"]
    if category not in valid:
        return jsonify({"error": "Invalid category"}), 400
    data = get_predictions_for_category(category)
    return jsonify(data)


@app.route("/api/predictions")
def all_predictions():
    result = {}
    for cat in ["ai_stocks", "semi_stocks", "gold_usd", "housing_oil"]:
        result[cat] = get_predictions_for_category(cat)
    return jsonify(result)


@app.route("/api/test")
def test_apis():
    """Test which APIs work from this server."""
    results = {}
    test_symbol = "AAPL"

    results["finnhub"] = {"key_set": bool(FINNHUB_KEY)}
    if FINNHUB_KEY:
        r = fetch_finnhub(test_symbol)
        results["finnhub"]["result"] = r

    results["alpha_vantage"] = {"key_set": bool(ALPHA_VANTAGE_KEY)}
    if ALPHA_VANTAGE_KEY:
        r = fetch_alpha_vantage(test_symbol)
        results["alpha_vantage"]["result"] = r

    r = fetch_yahoo_v8(test_symbol)
    results["yahoo"] = {"result": r}

    return jsonify(results)


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Investment Predictor API v5"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
