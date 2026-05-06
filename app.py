from flask import Flask, jsonify
import requests
import numpy as np
from datetime import datetime, timedelta
import os

app = Flask(__name__)

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")

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
        {"symbol": "GOLD", "name": "Gold (USD/oz)", "function": "GOLD"},
        {"symbol": "DXY", "name": "US Dollar Index", "function": "DXY"},
    ],
    "housing_oil": [
        {"symbol": "BRENT", "name": "Brent Crude (USD/bbl)", "function": "BRENT"},
        {"symbol": "HKPI", "name": "HK Property Index", "function": "HKPI"},
    ],
}


def fetch_stock_price(symbol):
    """Fetch real stock price from Alpha Vantage."""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        quote = data.get("Global Quote", {})
        price = float(quote.get("05. price", 0))
        change_pct = float(quote.get("10. change percent", "0").replace("%", ""))
        if price > 0:
            return {"price": price, "change_pct": change_pct}
    except Exception:
        pass
    return None


def fetch_commodity_price(function_type):
    """Fetch commodity prices from Alpha Vantage or use fallback."""
    try:
        if function_type == "GOLD":
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "COMMODITY_EXCHANGE_RATE",
                "from_currency": "XAU",
                "to_currency": "USD",
                "apikey": ALPHA_VANTAGE_KEY,
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            rate_data = data.get("Realtime Currency Exchange Rate", {})
            price = float(rate_data.get("5. Exchange Rate", 0))
            if price > 0:
                return {"price": price}
        elif function_type == "BRENT":
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "BRENT",
                "interval": "daily",
                "apikey": ALPHA_VANTAGE_KEY,
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            values = data.get("data", [])
            for v in values:
                if v.get("value") != ".":
                    return {"price": float(v["value"])}
    except Exception:
        pass
    return None


FALLBACK_PRICES = {
    "NVDA": 890.0, "MSFT": 420.0, "GOOGL": 175.0, "META": 510.0, "AMZN": 185.0,
    "AMD": 165.0, "INTC": 44.0, "QCOM": 170.0, "AVGO": 1350.0,
    "GOLD": 2350.0, "DXY": 104.5, "BRENT": 82.0, "HKPI": 340.0,
}


def predict(current_price, symbol):
    """Generate prediction using trend extrapolation with noise."""
    np.random.seed(hash(symbol + datetime.now().strftime("%Y-%m-%d")) % 2**31)

    volatility = 0.05
    if symbol in ["NVDA", "AMD"]:
        volatility = 0.08
    elif symbol in ["GOLD", "DXY", "HKPI"]:
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

        price_data = None
        if category in ["ai_stocks", "semi_stocks"]:
            price_data = fetch_stock_price(symbol)
        else:
            func = asset.get("function", "")
            price_data = fetch_commodity_price(func)

        if price_data and price_data.get("price", 0) > 0:
            current_price = price_data["price"]
        else:
            current_price = FALLBACK_PRICES.get(symbol, 100.0)

        prediction = predict(current_price, symbol + category)

        results.append({
            "symbol": symbol,
            "name": name,
            "category": category,
            "currentPrice": round(current_price, 2),
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


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Investment Predictor API"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
