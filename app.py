from flask import Flask, jsonify
import requests
import numpy as np
from datetime import datetime, timedelta
import os
import json
import xml.etree.ElementTree as ET
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history")
os.makedirs(DATA_DIR, exist_ok=True)
NEWS_CACHE_FILE = os.path.join(DATA_DIR, "news_cache.json")

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
    {"name": "NVIDIA", "query": "NVDA"},
    {"name": "Microsoft", "query": "MSFT"},
    {"name": "Google", "query": "GOOGL"},
    {"name": "Meta", "query": "META"},
    {"name": "OpenAI", "query": "OpenAI+Sam+Altman"},
    {"name": "Anthropic", "query": "Anthropic+Dario+Amodei+Claude"},
    {"name": "Amazon", "query": "AMZN"},
    {"name": "Intel", "query": "INTC"},
    {"name": "Qualcomm", "query": "QCOM"},
    {"name": "Samsung", "query": "005930.KS"},
    {"name": "Apple", "query": "AAPL"},
]

FALLBACK_PRICES = {
    "NVDA": 890.0, "MSFT": 420.0, "GOOGL": 175.0, "META": 510.0, "AMZN": 185.0,
    "AMD": 165.0, "INTC": 44.0, "QCOM": 170.0, "AVGO": 1350.0,
    "TSM": 160.0, "005930.KS": 75000.0, "MU": 120.0, "000660.KS": 180000.0,
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


PREDICTION_REASONS = {
    "NVDA": {
        "factors": ["GPU demand for AI training remains strong", "Data center revenue growth acceleration",
                    "CUDA ecosystem lock-in advantage", "Next-gen Blackwell architecture adoption"],
        "risks": ["High valuation multiple", "China export restrictions", "Customer concentration risk"]
    },
    "MSFT": {
        "factors": ["Azure AI services revenue growth", "Copilot enterprise adoption expanding",
                    "OpenAI partnership monetization", "Office 365 price increases"],
        "risks": ["AI investment CAPEX pressure on margins", "Antitrust scrutiny"]
    },
    "GOOGL": {
        "factors": ["Search AI integration (Gemini)", "Cloud revenue growth with AI workloads",
                    "YouTube ad revenue recovery", "Waymo autonomous driving progress"],
        "risks": ["Search market share pressure from AI chatbots", "DOJ antitrust ruling"]
    },
    "META": {
        "factors": ["Reels monetization improving", "AI-driven ad targeting efficiency",
                    "WhatsApp business API growth", "Llama open-source model ecosystem"],
        "risks": ["Reality Labs continued losses", "Regulatory pressure on data usage"]
    },
    "AMZN": {
        "factors": ["AWS AI/ML workload growth", "Retail margin improvement",
                    "Advertising business acceleration", "Alexa AI upgrade potential"],
        "risks": ["Retail competition", "Heavy CAPEX spending cycle"]
    },
    "AMD": {
        "factors": ["MI300X AI accelerator ramp", "Data center GPU market share gains",
                    "EPYC server CPU momentum", "AI PC chip cycle"],
        "risks": ["NVIDIA competitive moat", "Inventory correction risk"]
    },
    "INTC": {
        "factors": ["Foundry services progress (18A node)", "Government CHIPS Act funding",
                    "AI PC refresh cycle", "Gaudi AI accelerator traction"],
        "risks": ["Execution risk on node transitions", "Market share loss continues", "Cash burn from foundry buildout"]
    },
    "QCOM": {
        "factors": ["Snapdragon X Elite for AI PCs", "On-device AI inference leadership",
                    "Automotive chip design wins", "IoT diversification"],
        "risks": ["Smartphone market cyclicality", "Apple modem transition", "ARM license dispute"]
    },
    "AVGO": {
        "factors": ["Custom AI accelerator demand (Google TPU)", "VMware integration synergies",
                    "Networking silicon for AI clusters", "Stable dividend growth"],
        "risks": ["Customer concentration", "Integration execution risk"]
    },
    "TSM": {
        "factors": ["3nm/2nm capacity ramp for AI chips", "Sole advanced node manufacturer",
                    "CoWoS packaging demand surge", "Global fab diversification (Arizona, Japan)"],
        "risks": ["Geopolitical Taiwan risk", "CAPEX cycle intensity", "Customer demand volatility"]
    },
    "005930.KS": {
        "factors": ["HBM3E memory for AI servers", "Memory cycle upturn",
                    "Foundry 2nm GAA technology", "Galaxy AI device ecosystem"],
        "risks": ["HBM yield challenges vs SK Hynix", "China market uncertainty", "Won currency volatility"]
    },
    "MU": {
        "factors": ["HBM demand from AI GPU makers", "Memory pricing recovery",
                    "Data center DRAM growth", "NAND supply discipline"],
        "risks": ["Memory cycle volatility", "China ban on certain products", "CAPEX timing"]
    },
    "000660.KS": {
        "factors": ["HBM market leader (supplying NVIDIA)", "Memory price upcycle",
                    "Advanced packaging technology lead", "AI server DRAM content growth"],
        "risks": ["Capacity expansion costs", "Technology competition from Samsung", "Won currency risk"]
    },
    "GC=F": {
        "factors": ["Central bank gold buying continues", "Geopolitical uncertainty hedge",
                    "Real interest rate trajectory", "De-dollarization trend"],
        "risks": ["Strong USD pressure", "Risk-on market sentiment shift", "ETF outflows"]
    },
    "CNY=X": {
        "factors": ["US-China yield differential", "PBoC policy direction",
                    "Trade balance dynamics", "Capital flow trends"],
        "risks": ["Fed rate path uncertainty", "China economic slowdown", "Geopolitical tensions"]
    },
}

DEFAULT_REASONS = {
    "factors": ["Technical momentum signals", "Sector trend alignment", "Historical pattern matching"],
    "risks": ["Market volatility", "Macro uncertainty"]
}


def get_prediction_reasons(symbol, trend):
    """Get prediction reasoning for a symbol."""
    reasons = PREDICTION_REASONS.get(symbol, DEFAULT_REASONS)
    factors = reasons["factors"]
    risks = reasons["risks"]

    if trend == "UP":
        summary = "Bullish outlook driven by strong fundamental catalysts"
    elif trend == "DOWN":
        summary = "Cautious outlook due to elevated risk factors"
    else:
        summary = "Neutral stance with balanced bull/bear factors"

    return {
        "summary": summary,
        "bullFactors": factors,
        "riskFactors": risks,
    }


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


def fetch_asset_data(asset, category, cny_rate, today_str):
    """Fetch data for a single asset (used in parallel)."""
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
    reasons = get_prediction_reasons(symbol, prediction["trend"])

    save_daily_prediction(symbol, today_str, predicted_prices)

    comparison = build_comparison(symbol, history_prices, history_dates, predicted_prices)

    return {
        "symbol": display_symbol,
        "name": name,
        "category": category,
        "currentPrice": round(current_price, 2),
        "source": source,
        "historyPrices": history_prices,
        "predictedPrices": predicted_prices,
        "comparison": comparison,
        "reasons": reasons,
        **prediction,
    }


def get_predictions_for_category(category):
    assets = ASSETS.get(category, [])
    today_str = datetime.now().strftime("%Y-%m-%d")

    cny_rate = None
    if category == "gold_usd":
        cny_rate = get_cny_rate()

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_asset_data, asset, category, cny_rate, today_str): asset
            for asset in assets
        }
        for future in as_completed(futures):
            try:
                result = future.result(timeout=20)
                results.append(result)
            except Exception as e:
                asset = futures[future]
                print(f"Error fetching {asset['symbol']}: {e}")

    order = [a.get("display", a["symbol"]) for a in assets]
    results.sort(key=lambda x: order.index(x["symbol"]) if x["symbol"] in order else 999)

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
    if pub:
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
    """Fetch news using Yahoo Finance search API (same domain as price API)."""
    items = []

    # Use Yahoo Finance search API - same domain as price fetching, confirmed working
    try:
        symbol = query.split("+")[0]
        url = f"https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        params = {"q": symbol, "newsCount": num, "quotesCount": 0}
        resp = requests.get(url, headers=headers, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            news_list = data.get("news", [])
            for n in news_list[:num]:
                title = n.get("title", "")
                link = n.get("link", "")
                pub_ts = n.get("providerPublishTime", 0)
                try:
                    pub_date = datetime.fromtimestamp(int(pub_ts)).strftime("%a, %d %b %Y %H:%M") if pub_ts else ""
                except:
                    pub_date = ""
                source = n.get("publisher", "")
                items.append({"title": title, "link": link, "pubDate": pub_date, "source": source})
    except Exception as e:
        print(f"Yahoo search news error for {query}: {e}")

    # Fallback to Google News RSS
    if not items:
        try:
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:num]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    source = item.find("source").text if item.find("source") is not None else ""
                    items.append({"title": title, "link": link, "pubDate": pub_date, "source": source})
        except Exception as e:
            print(f"Google News error for {query}: {e}")

    return items


CHINESE_AI_SOURCES = [
    {
        "name": "机器之心",
        "url": "https://www.jiqizhixin.com/",
        "type": "jiqizhixin",
    },
    {
        "name": "量子位",
        "url": "https://www.qbitai.com/",
        "type": "qbitai",
    },
    {
        "name": "雷峰网",
        "url": "https://www.leiphone.com/category/ai",
        "type": "leiphone",
    },
    {
        "name": "爱范儿",
        "url": "https://www.ifanr.com/category/ai",
        "type": "ifanr",
    },
    {
        "name": "品玩",
        "url": "https://www.pingwest.com/",
        "type": "pingwest",
    },
    {
        "name": "极客公园",
        "url": "https://www.geekpark.net/",
        "type": "geekpark",
    },
]

HEADERS_ZH = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def fetch_jiqizhixin():
    """Fetch AI articles from 机器之心 via RSS or HTML."""
    items = []
    try:
        # Try RSS first
        resp = requests.get("https://www.jiqizhixin.com/rss", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200 and "xml" in resp.headers.get("content-type", ""):
            root = ET.fromstring(resp.content)
            for item in root.findall(".//{http://www.w3.org/2005/Atom}entry")[:8]:
                title_el = item.find("{http://www.w3.org/2005/Atom}title")
                link_el = item.find("{http://www.w3.org/2005/Atom}link")
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                link = link_el.get("href", "") if link_el is not None else ""
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "机器之心", "pubDate": "", "company": "AI"})
            if items:
                return items
        # Fallback: HTML scraping
        resp = requests.get("https://www.jiqizhixin.com/", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            # Try multiple patterns
            articles = re.findall(r'href="(/articles/[^"]+)"[^>]*>([^<]{5,})</a>', resp.text)
            if not articles:
                articles = re.findall(r'<a[^>]+href="(/articles/[^"]+)"[^>]*title="([^"]+)"', resp.text)
            if not articles:
                articles = re.findall(r'"url":"(/articles/[^"]+)","title":"([^"]+)"', resp.text)
            for path, title in articles[:8]:
                title = title.strip()
                if title and len(title) > 4:
                    link = f"https://www.jiqizhixin.com{path}" if path.startswith("/") else path
                    items.append({"title": title, "link": link, "source": "机器之心", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"jiqizhixin error: {e}")
    return items


def fetch_qbitai():
    """Fetch AI articles from 量子位."""
    items = []
    try:
        resp = requests.get("https://www.qbitai.com/", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'<a[^>]+href="(https?://www\.qbitai\.com/\d+/\d+/[^"]+)"[^>]*>([^<]+)</a>', resp.text)
            if not articles:
                articles = re.findall(r'href="(https?://www\.qbitai\.com/[^"]*\d+\.html)"[^>]*>([^<]+)</a>', resp.text)
            for link, title in articles[:8]:
                title = title.strip()
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "量子位", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"qbitai error: {e}")
    return items


def fetch_leiphone():
    """Fetch AI articles from 雷峰网."""
    items = []
    try:
        resp = requests.get("https://www.leiphone.com/category/ai", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'<a[^>]+href="(https?://www\.leiphone\.com/[^"]+)"[^>]*title="([^"]+)"', resp.text)
            if not articles:
                articles = re.findall(r'href="(https?://www\.leiphone\.com/category/ai/[^"]+)"[^>]*>([^<]{5,})</a>', resp.text)
            for link, title in articles[:8]:
                title = title.strip()
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "雷峰网", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"leiphone error: {e}")
    return items


def fetch_ifanr():
    """Fetch AI articles from 爱范儿 via RSS feed."""
    items = []
    try:
        # ifanr is WordPress, try RSS feed
        resp = requests.get("https://www.ifanr.com/feed", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            try:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:10]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    title = title.strip() if title else ""
                    if title and len(title) > 4:
                        items.append({"title": title, "link": link, "source": "爱范儿", "pubDate": pub_date, "company": "AI"})
                if items:
                    return items
            except ET.ParseError:
                pass
        # Fallback: HTML
        resp = requests.get("https://www.ifanr.com/category/ai", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'<a[^>]+href="(https?://www\.ifanr\.com/\d+)"[^>]*>([^<]{5,})</a>', resp.text)
            for link, title in articles[:8]:
                title = title.strip()
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "爱范儿", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"ifanr error: {e}")
    return items


def fetch_pingwest():
    """Fetch AI articles from 品玩 via API or HTML."""
    items = []
    try:
        # Try PingWest API
        resp = requests.get("https://api.pingwest.com/a/list/1?pagesize=10", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            try:
                data = resp.json()
                article_list = data.get("data", {}).get("list", [])
                if not article_list:
                    article_list = data.get("list", [])
                for art in article_list[:8]:
                    title = art.get("title", "").strip()
                    art_id = art.get("id", "")
                    link = f"https://www.pingwest.com/a/{art_id}" if art_id else ""
                    if title and len(title) > 4 and link:
                        items.append({"title": title, "link": link, "source": "品玩", "pubDate": "", "company": "AI"})
                if items:
                    return items
            except:
                pass
        # Fallback: HTML
        resp = requests.get("https://www.pingwest.com/", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'href="((?:https?://www\.pingwest\.com)?/a/\d+)"[^>]*>([^<]{5,})</a>', resp.text)
            for link, title in articles[:8]:
                title = title.strip()
                if not link.startswith("http"):
                    link = f"https://www.pingwest.com{link}"
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "品玩", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"pingwest error: {e}")
    return items


def fetch_geekpark():
    """Fetch AI articles from 极客公园 via RSS or HTML."""
    items = []
    try:
        # Try RSS
        resp = requests.get("https://www.geekpark.net/rss", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            try:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:10]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    title = title.strip() if title else ""
                    if title and len(title) > 4:
                        items.append({"title": title, "link": link, "source": "极客公园", "pubDate": pub_date, "company": "AI"})
                if items:
                    return items
            except ET.ParseError:
                pass
        # Fallback: HTML with multiple regex patterns
        resp = requests.get("https://www.geekpark.net/", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'href="((?:https?://www\.geekpark\.net)?/news/\d+)"[^>]*>([^<]{5,})</a>', resp.text)
            if not articles:
                articles = re.findall(r'"url":"(/news/\d+)","title":"([^"]+)"', resp.text)
            for link, title in articles[:8]:
                title = title.strip()
                if not link.startswith("http"):
                    link = f"https://www.geekpark.net{link}"
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "极客公园", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"geekpark error: {e}")
    return items


def fetch_36kr():
    """Fetch AI articles from 36氪."""
    items = []
    try:
        resp = requests.get("https://36kr.com/information/AI/", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'href="(/p/\d+)"[^>]*>([^<]{5,})</a>', resp.text)
            if not articles:
                articles = re.findall(r'"route":"/p/(\d+)"[^}]*"title":"([^"]+)"', resp.text)
                articles = [(f"/p/{aid}", t) for aid, t in articles]
            for path, title in articles[:8]:
                title = title.strip()
                link = f"https://36kr.com{path}" if path.startswith("/") else path
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "36氪", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"36kr error: {e}")
    return items


def fetch_ithome():
    """Fetch AI articles from IT之家."""
    items = []
    try:
        resp = requests.get("https://www.ithome.com/tag/AI", headers=HEADERS_ZH, timeout=10)
        if resp.status_code == 200:
            import re
            articles = re.findall(r'<a[^>]+href="(https?://www\.ithome\.com/0/\d+/\d+\.htm)"[^>]*>([^<]{5,})</a>', resp.text)
            if not articles:
                articles = re.findall(r'href="(https?://www\.ithome\.com/\d+/\d+/\d+\.htm)"[^>]*>([^<]{5,})</a>', resp.text)
            for link, title in articles[:8]:
                title = title.strip()
                if title and len(title) > 4:
                    items.append({"title": title, "link": link, "source": "IT之家", "pubDate": "", "company": "AI"})
    except Exception as e:
        print(f"ithome error: {e}")
    return items


AI_KEYWORDS = [
    "AI", "ai", "人工智能", "大模型", "GPT", "gpt", "LLM", "机器学习", "深度学习",
    "神经网络", "智能体", "Agent", "agent", "ChatGPT", "Copilot", "算力", "芯片",
    "英伟达", "NVIDIA", "nvidia", "半导体", "自动驾驶", "智驾", "大语言模型",
    "Transformer", "transformer", "生成式", "AIGC", "Sora", "Claude", "Gemini",
    "OpenAI", "Anthropic", "DeepMind", "智算", "训练", "推理", "模型",
    "机器人", "具身智能", "多模态", "RAG", "向量", "Llama", "通义", "文心",
    "豆包", "Kimi", "智谱", "百川", "讯飞", "星火",
]


def is_ai_related(title):
    """Check if article title is AI-related."""
    return any(kw in title for kw in AI_KEYWORDS)


def fetch_all_chinese_ai_news():
    """Fetch AI news from all Chinese tech media sources in parallel."""
    fetchers = [
        fetch_jiqizhixin,
        fetch_qbitai,
        fetch_leiphone,
        fetch_ifanr,
        fetch_pingwest,
        fetch_geekpark,
        fetch_36kr,
        fetch_ithome,
    ]
    all_news = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fn): fn.__name__ for fn in fetchers}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_news.extend(result)
                print(f"{futures[future]}: got {len(result)} articles")
            except Exception as e:
                print(f"{futures[future]} failed: {e}")

    seen_titles = set()
    unique_news = []
    for item in all_news:
        if item["title"] not in seen_titles and is_ai_related(item["title"]):
            seen_titles.add(item["title"])
            unique_news.append(item)

    return unique_news[:30]


def get_ai_news(lang="en"):
    """Get AI company news, with caching (refresh every 4 hours)."""
    cache_file = NEWS_CACHE_FILE if lang == "en" else NEWS_CACHE_FILE.replace(".json", f"_{lang}.json")
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
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

    if not needs_refresh and "news" in cache and len(cache["news"]) > 0:
        return cache["news"]

    all_news = []

    if lang == "zh":
        all_news = fetch_all_chinese_ai_news()
    else:
        companies = AI_NEWS_COMPANIES

        def fetch_company_news(company):
            try:
                items = fetch_google_news_rss(company["query"], num=3)
                for item in items:
                    item["company"] = company["name"]
                return items
            except Exception as e:
                print(f"Error fetching news for {company['name']}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_company_news, c): c for c in companies}
            for future in as_completed(futures):
                all_news.extend(future.result())

    try:
        all_news.sort(key=lambda x: news_importance_score(x), reverse=True)
    except:
        pass

    # Only cache if we got results
    if all_news:
        cache = {
            "last_update": now.strftime("%Y-%m-%d %H:%M"),
            "news": all_news,
        }
        try:
            with open(cache_file, "w") as f:
                json.dump(cache, f, ensure_ascii=False)
        except:
            pass

    return all_news


@app.route("/api/news")
def ai_news():
    """Get AI company news. Use ?lang=zh for Chinese."""
    try:
        from flask import request
        lang = request.args.get("lang", "en")

        # Delete stale cache to force fresh fetch
        cache_file = NEWS_CACHE_FILE if lang == "en" else NEWS_CACHE_FILE.replace(".json", f"_{lang}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
                if not cache.get("news"):
                    os.remove(cache_file)
            except:
                if os.path.exists(cache_file):
                    os.remove(cache_file)

        news = get_ai_news(lang=lang)
        return jsonify(news)
    except Exception as e:
        print(f"News endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/test")
def test_apis():
    results = {}
    r = fetch_yahoo_v8("AAPL")
    results["yahoo"] = {"result": r}
    cny = get_cny_rate()
    results["cny_rate"] = cny

    # Test news fetching
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        params = {"q": "NVDA", "newsCount": 3, "quotesCount": 0}
        resp = requests.get(url, headers=headers, params=params, timeout=8)
        results["news_test"] = {
            "status": resp.status_code,
            "has_news": "news" in resp.json() if resp.status_code == 200 else False,
            "news_count": len(resp.json().get("news", [])) if resp.status_code == 200 else 0,
            "raw_keys": list(resp.json().keys()) if resp.status_code == 200 else [],
        }
    except Exception as e:
        results["news_test"] = {"error": str(e)}

    return jsonify(results)


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Investment Predictor API v8"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
