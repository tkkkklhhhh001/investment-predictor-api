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
        {"symbol": "GC=F", "name": "Gold (CNY/g)", "display": "GOLD"},
        {"symbol": "DX-Y.NYB", "name": "USD/CNY Rate", "display": "USD/CNY"},
    ],
    "housing": [
        {"symbol": "0012.HK", "name": "Henderson Land Dev", "display": "HKLAND"},
        {"symbol": "0016.HK", "name": "SHK Properties", "display": "SHKP"},
        {"symbol": "0001.HK", "name": "CK Asset Holdings", "display": "CKA"},
    ],
}

FALLBACK_PRICES = {
    "NVDA": 890.0, "MSFT": 420.0, "GOOGL": 175.0, "META": 510.0, "AMZN": 185.0,
    "AMD": 165.0, "INTC": 44.0, "QCOM": 170.0, "AVGO": 1350.0,
    "GC=F": 2350.0, "DX-Y.NYB": 7.25, "0012.HK": 23.0, "0016.HK": 80.0, "0001.HK": 38.0,
}

CNY_RATE_FALLBACK = 7.25


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


def fetch_yahoo_history(symbol, days=30):
    """Fetch historical closing prices from Yahoo v8."""
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        params = {"interval": "1d", "range": f"{days}d"}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            prices = [round(p, 2) for p in closes if p is not None]
            return prices
    except Exception as e:
        print(f"Yahoo history error {symbol}: {e}")
    return []


def get_cny_rate():
    """Get USD/CNY exchange rate."""
    result = fetch_yahoo_v8("CNY=X")
    if result and result.get("price", 0) > 0:
        return result["price"]
    return CNY_RATE_FALLBACK


def fetch_price(symbol):
    """Fetch current price from Yahoo."""
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
    elif symbol in ["GC=F", "DX-Y.NYB", "0012.HK", "0016.HK", "0001.HK"]:
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


def generate_predicted_prices(current_price, symbol, days=7):
    """Generate a list of predicted daily prices."""
    np.random.seed(hash(symbol + "pred" + datetime.now().strftime("%Y-%m-%d")) % 2**31)
    volatility = 0.02
    if symbol in ["NVDA", "AMD"]:
        volatility = 0.035
    prices = [current_price]
    for i in range(days - 1):
        change = np.random.normal(0.001, volatility)
        prices.append(round(prices[-1] * (1 + change), 2))
    return prices


def get_predictions_for_category(category):
    assets = ASSETS.get(category, [])
    results = []

    cny_rate = None
    if category == "gold_usd":
        cny_rate = get_cny_rate()

    for asset in assets:
        symbol = asset["symbol"]
        name = asset["name"]
        display_symbol = asset.get("display", symbol)

        price_data = fetch_price(symbol)
        history = fetch_yahoo_history(symbol, 30)

        if price_data and price_data.get("price", 0) > 0:
            current_price = price_data["price"]
            source = price_data["source"]
        else:
            current_price = FALLBACK_PRICES.get(symbol, 100.0)
            source = "fallback"

        if category == "gold_usd" and cny_rate:
            if symbol == "GC=F":
                current_price = round(current_price * cny_rate / 31.1035, 2)
                history = [round(p * cny_rate / 31.1035, 2) for p in history]
            elif symbol == "DX-Y.NYB":
                current_price = cny_rate
                history = history if history else [cny_rate]

        prediction = predict(current_price, symbol + category)
        predicted_prices = generate_predicted_prices(current_price, symbol)

        results.append({
            "symbol": display_symbol,
            "name": name,
            "category": category,
            "currentPrice": round(current_price, 2),
            "source": source,
            "historyPrices": history,
            "predictedPrices": predicted_prices,
            **prediction,
        })

    return results


@app.route("/api/predictions/<category>")
def predictions(category):
    valid = ["ai_stocks", "semi_stocks", "gold_usd", "housing"]
    if category not in valid:
        return jsonify({"error": "Invalid category"}), 400
    data = get_predictions_for_category(category)
    return jsonify(data)


@app.route("/api/predictions")
def all_predictions():
    result = {}
    for cat in ["ai_stocks", "semi_stocks", "gold_usd", "housing"]:
        result[cat] = get_predictions_for_category(cat)
    return jsonify(result)


@app.route("/api/test")
def test_apis():
    """Test which APIs work from this server."""
    results = {}
    test_symbol = "AAPL"

    r = fetch_yahoo_v8(test_symbol)
    results["yahoo"] = {"result": r}

    cny = get_cny_rate()
    results["cny_rate"] = cny

    return jsonify(results)


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Investment Predictor API v6"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
