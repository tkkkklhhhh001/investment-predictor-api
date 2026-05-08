from flask import Flask, jsonify
import requests
import numpy as np
from datetime import datetime, timedelta
import os
import json
import xml.etree.ElementTree as ET
import re
import hashlib

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history")
os.makedirs(DATA_DIR, exist_ok=True)

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
        {"symbol": "TSM", "name": "TSMC"},
        {"symbol": "005930.KS", "name": "Samsung", "display": "SAMSUNG"},
        {"symbol": "MU", "name": "Micron"},
        {"symbol": "000660.KS", "name": "SK Hynix", "display": "SKHYNIX"},
    ],
    "gold_usd": [
        {"symbol": "GC=F", "name": "Gold (CNY/g)", "display": "GOLD"},
        {"symbol": "CNY=X", "name": "USD/CNY Rate", "display": "USD/CNY"},
    ],
}

AI_NEWS_COMPANIES = [
    {"name": "NVIDIA", "query": "NVIDIA+Jensen+Huang"},
    {"name": "Microsoft", "query": "Microsoft+Satya+Nadella+AI"},
    {"name": "Google", "query": "Google+DeepMind+Sundar+Pichai+AI"},
    {"name": "Meta", "query": "Meta+Mark+Zuckerberg+AI"},
    {"name": "OpenAI", "query": "OpenAI+Sam+Altman"},
    {"name": "Anthropic", "query": "Anthropic+Dario+Amodei+Claude"},
    {"name": "Amazon", "query": "Amazon+Andy+Jassy+AI"},
    {"name": "Intel", "query": "Intel+Pat+Gelsinger+AI+chip"},
    {"name": "Qualcomm", "query": "Qualcomm+Cristiano+Amon+AI"},
    {"name": "Samsung", "query": "Samsung+semiconductor+AI+chip"},
    {"name": "Apple", "query": "Apple+Tim+Cook+AI"},
]

FALLBACK_PRICES = {
    "NVDA": 890.0, "MSFT": 420.0, "GOOGL": 175.0, "META": 510.0, "AMZN": 185.0,
    "AMD": 165.0, "INTC": 44.0, "QCOM": 170.0, "AVGO": 1350.0,
    "GC=F": 2350.0, "CNY=X": 7.25,
}

CNY_RATE_FALLBACK = 7.25


def fetch_yahoo_v8(symbol):
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        params = {"interval": "1d", "range": f"{days}d"}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            timestamps = result[0].get("timestamp", [])
            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            prices = []
            dates = []
            for i, p in enumerate(closes):
                if p is not None:
                    prices.append(round(p, 2))
                    if i < len(timestamps):
                        dates.append(datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d"))
            return prices, dates
    except Exception as e:
        print(f"Yahoo history error {symbol}: {e}")
    return [], []


def get_cny_rate():
    result = fetch_yahoo_v8("CNY=X")
    if result and result.get("price", 0) > 0:
        return result["price"]
    return CNY_RATE_FALLBACK


def fetch_price(symbol):
    result = fetch_yahoo_v8(symbol)
    if result:
        return result
    return None


def generate_predicted_prices(current_price, symbol, days=30):
    np.random.seed(hash(symbol + "pred" + datetime.now().strftime("%Y-%m-%d")) % 2**31)
    volatility = 0.02
    if symbol in ["NVDA", "AMD"]:
        volatility = 0.035
    prices = [current_price]
    for i in range(days - 1):
        change = np.random.normal(0.001, volatility)
        prices.append(round(prices[-1] * (1 + change), 2))
    return prices


def save_daily_prediction(symbol, date_str, predicted_prices):
    """Save today's prediction to disk so we can compare later."""
    filepath = os.path.join(DATA_DIR, f"{symbol.replace('=', '_').replace('.', '_')}.json")
    history = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                history = json.load(f)
        except:
            history = {}
    if date_str not in history:
        history[date_str] = predicted_prices
        with open(filepath, "w") as f:
            json.dump(history, f)


def load_prediction_history(symbol):
    """Load all saved predictions for a symbol."""
    filepath = os.path.join(DATA_DIR, f"{symbol.replace('=', '_').replace('.', '_')}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def build_comparison(symbol, history_prices, history_dates, predicted_prices):
    """
    Build comparison data:
    - actualPrices: real historical prices (last 30d)
    - predictedHistoryPrices: what we predicted for those dates (from saved predictions)
    - futurePredictedPrices: today's prediction for next 7 days
    """
    pred_history = load_prediction_history(symbol)

    predicted_for_past = []
    for i, date_str in enumerate(history_dates):
        found = False
        for pred_date, pred_prices in pred_history.items():
            pred_dt = datetime.strptime(pred_date, "%Y-%m-%d")
            target_dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_diff = (target_dt - pred_dt).days
            if 0 <= day_diff < len(pred_prices):
                predicted_for_past.append(pred_prices[day_diff])
                found = True
                break
        if not found:
            predicted_for_past.append(None)

    return {
        "actualPrices": history_prices,
        "actualDates": history_dates,
        "predictedHistoryPrices": predicted_for_past,
        "futurePredictedPrices": predicted_prices,
    }


def predict(current_price, symbol):
    np.random.seed(hash(symbol + datetime.now().strftime("%Y-%m-%d")) % 2**31)

    volatility = 0.05
    if symbol in ["NVDA", "AMD"]:
        volatility = 0.08
    elif symbol in ["GC=F", "CNY=X", "0012.HK", "0016.HK", "0001.HK"]:
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
    today_str = datetime.now().strftime("%Y-%m-%d")

    cny_rate = None
    if category == "gold_usd":
        cny_rate = get_cny_rate()

    for asset in assets:
        symbol = asset["symbol"]
        name = asset["name"]
        display_symbol = asset.get("display", symbol)

        price_data = fetch_price(symbol)
        history_prices, history_dates = fetch_yahoo_history(symbol, 30)

        if price_data and price_data.get("price", 0) > 0:
            current_price = price_data["price"]
            source = price_data["source"]
        else:
            current_price = FALLBACK_PRICES.get(symbol, 100.0)
            source = "fallback"

        if category == "gold_usd" and cny_rate:
            if symbol == "GC=F":
                current_price = round(current_price * cny_rate / 31.1035, 2)
                history_prices = [round(p * cny_rate / 31.1035, 2) for p in history_prices]

        prediction = predict(current_price, symbol + category)
        predicted_prices = generate_predicted_prices(current_price, symbol)

        save_daily_prediction(symbol, today_str, predicted_prices)

        comparison = build_comparison(symbol, history_prices, history_dates, predicted_prices)

        results.append({
            "symbol": display_symbol,
            "name": name,
            "category": category,
            "currentPrice": round(current_price, 2),
            "source": source,
            "historyPrices": history_prices,
            "predictedPrices": predicted_prices,
            "comparison": comparison,
            **prediction,
        })

    return results


@app.route("/api/predictions/<category>")
def predictions(category):
    valid = ["ai_stocks", "semi_stocks", "gold_usd"]
    if category not in valid:
        return jsonify({"error": "Invalid category"}), 400
    data = get_predictions_for_category(category)
    return jsonify(data)


@app.route("/api/predictions")
def all_predictions():
    result = {}
    for cat in ["ai_stocks", "semi_stocks", "gold_usd"]:
        result[cat] = get_predictions_for_category(cat)
    return jsonify(result)


# --- AI News Feature ---

MAJOR_SOURCES = ["Reuters", "Bloomberg", "CNBC", "The Wall Street Journal", "Financial Times",
                  "The New York Times", "TechCrunch", "The Verge", "Wired", "Associated Press"]
CXO_NAMES = ["Jensen Huang", "Satya Nadella", "Sundar Pichai", "Mark Zuckerberg", "Sam Altman",
             "Dario Amodei", "Tim Cook", "Pat Gelsinger", "Cristiano Amon", "Andy Jassy", "Lisa Su"]


def news_importance_score(item):
    """Score news by importance: source reputation + CXO mention + recency."""
    score = 0
    title = item.get("title", "")
    source = item.get("source", "")

    if any(s.lower() in source.lower() for s in MAJOR_SOURCES):
        score += 50

    if any(name.lower() in title.lower() for name in CXO_NAMES):
        score += 40

    keywords = ["CEO", "CTO", "announce", "launch", "billion", "acquire", "partnership",
                "breakthrough", "record", "earnings", "revenue"]
    for kw in keywords:
        if kw.lower() in title.lower():
            score += 10
            break

    pub = item.get("pubDate", "")
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub)
        hours_ago = (datetime.now(dt.tzinfo) - dt).total_seconds() / 3600
        if hours_ago < 6:
            score += 30
        elif hours_ago < 24:
            score += 20
        elif hours_ago < 48:
            score += 10
    except:
        pass

    return score




def fetch_google_news_rss(query, num=5):
    """Fetch news from Google News RSS."""
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:num]:
            title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            source = item.find("source").text if item.find("source") is not None else ""
            items.append({
                "title": title,
                "link": link,
                "pubDate": pub_date,
                "source": source,
            })
        return items
    except Exception as e:
        print(f"News fetch error for {query}: {e}")
        return []


def get_ai_news():
    """Get AI company news, with caching (refresh every 4 hours)."""
    cache = {}
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except:
            cache = {}

    last_update = cache.get("last_update", "")
    now = datetime.now()
    needs_refresh = True

    if last_update:
        try:
            last_dt = datetime.strptime(last_update, "%Y-%m-%d %H:%M")
            if (now - last_dt).total_seconds() < 4 * 3600:
                needs_refresh = False
        except:
            pass

    if not needs_refresh and "news" in cache:
        return cache["news"]

    all_news = []
    for company in AI_NEWS_COMPANIES:
        items = fetch_google_news_rss(company["query"], num=3)
        for item in items:
            item["company"] = company["name"]
        all_news.extend(items)

    all_news.sort(key=lambda x: news_importance_score(x), reverse=True)

    cache = {
        "last_update": now.strftime("%Y-%m-%d %H:%M"),
        "news": all_news,
    }
    try:
        with open(NEWS_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except:
        pass

    return all_news


@app.route("/api/news")
def ai_news():
    """Get AI company news."""
    news = get_ai_news()
    return jsonify(news)


@app.route("/api/test")
def test_apis():
    results = {}
    r = fetch_yahoo_v8("AAPL")
    results["yahoo"] = {"result": r}
    cny = get_cny_rate()
    results["cny_rate"] = cny
    return jsonify(results)


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Investment Predictor API v8"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
