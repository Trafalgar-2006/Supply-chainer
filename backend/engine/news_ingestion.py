import calendar

import feedparser
import requests
import urllib.parse
import time
from typing import Optional

FEED_TIMEOUT_S = 2.0
NEWS_MAX_AGE_S = 3 * 24 * 3600  # older headlines describe events that may be over

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
# WMO weather codes (open-meteo.com/en/docs). Fog, drizzle, rain, snow and showers
# slow freight ("rainy" for the delay model); thunderstorms, heavy snow and
# violent showers stop it ("stormy"), as do gale-force winds.
STORMY_CODES = {65, 67, 75, 82, 86, 95, 96, 99}
WET_CODES = {45, 48, 51, 53, 55, 56, 57, 61, 63, 66, 71, 73, 77, 80, 81, 85}
GALE_KMH = 62  # Beaufort 8
WEATHER_WORDS = [((0,), "clear"), ((1, 2, 3), "cloudy"), ((45, 48), "fog"), ((51, 53, 55, 56, 57), "drizzle"),
                 ((61, 63, 65, 66, 67), "rain"), ((71, 73, 75, 77), "snow"), ((80, 81, 82), "rain showers"),
                 ((85, 86), "snow showers"), ((95, 96, 99), "thunderstorm")]


def describe_weather(code: int) -> str:
    return next((word for codes, word in WEATHER_WORDS if code in codes), "unknown")

class DynamicNewsIngestor:
    """
    Supplychainer Stage 1: Dynamic News Ingestion.
    Consumes live external intelligence from Google News RSS.
    Implements location-aware text retrieval and caching.
    """
    def __init__(self):
        self.cache = {} # {(query, max_items): (timestamp, content or None)}
        self.cache_ttl = 900 # 15 minutes
        self.failure_ttl = 300 # retry an unreachable feed after five minutes, not on every request

        # Operational Physics Fallbacks (Offline Reliability)
        self.fallback_news = {
            "sea": "Maritime congestion reported at major transshipment hubs. Berthing delays expected.",
            "air": "Aviation fuel surcharge volatility and cargo handling backlogs noted in international airports.",
            "road": "Highway traffic density increasing in primary logistics corridors.",
            "rail": "Rail freight scheduling adjustments due to infrastructure maintenance."
        }

    def fetch_headlines(self, location: str, max_items: int = 8) -> Optional[str]:
        """Recent logistics-disruption headlines for a place, newest first, or None.

        Only headlines from the last three days count; the feed's search results
        can be months old. None means "no live signal", which callers must not
        confuse with the static fallback reports.
        """
        query = f"{location} logistics disruption"
        key = (query, max_items)
        now = time.time()
        if key in self.cache:
            ts, content = self.cache[key]
            if now - ts < (self.cache_ttl if content else self.failure_ttl):
                return content

        content = None
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            # Per-request timeout; socket.setdefaulttimeout would change every socket in the process.
            response = requests.get(url, timeout=FEED_TIMEOUT_S)
            response.raise_for_status()
            recent = sorted((e for e in feedparser.parse(response.content).entries
                             if e.get("published_parsed") and now - calendar.timegm(e.published_parsed) <= NEWS_MAX_AGE_S),
                            key=lambda e: e.published_parsed, reverse=True)
            if recent:
                content = " | ".join(entry.title for entry in recent[:max_items])
        except Exception as e:
            print(f"[NEWS] Feed unavailable for {location!r}: {e}")
        self.cache[key] = (now, content)
        return content

    def fetch_weather(self, lat: float, lon: float) -> Optional[dict]:
        """Current weather at a point from Open-Meteo (no key; CC BY 4.0), or None if unavailable.

        Returns the delay model's condition ("clear", "rainy" or "stormy") with the
        readings it came from.
        """
        key = ("weather", round(lat, 1), round(lon, 1))
        now = time.time()
        if key in self.cache:
            ts, content = self.cache[key]
            if now - ts < (self.cache_ttl if content else self.failure_ttl):
                return content

        content = None
        try:
            response = requests.get(WEATHER_URL, timeout=FEED_TIMEOUT_S, params={
                "latitude": round(lat, 3), "longitude": round(lon, 3),
                "current": "weather_code,wind_speed_10m,precipitation"})
            response.raise_for_status()
            current = response.json()["current"]
            code, wind = int(current["weather_code"]), float(current["wind_speed_10m"])
            precipitation = float(current["precipitation"])
            if code in STORMY_CODES or wind >= GALE_KMH:
                condition = "stormy"
            elif code in WET_CODES or precipitation >= 0.5:
                condition = "rainy"
            else:
                condition = "clear"
            content = {"condition": condition, "description": describe_weather(code),
                       "wind_kmh": round(wind), "precipitation_mm": round(precipitation, 1)}
        except Exception as e:
            print(f"[WEATHER] Unavailable for ({lat:.1f}, {lon:.1f}): {e}")
        self.cache[key] = (now, content)
        return content

    def get_latest_news(self, location: str, transport_mode: str) -> str:
        """Live headlines for a location, or the mode's static fallback report."""
        return self.fetch_headlines(location) or self.fallback_news.get(
            transport_mode.lower(), "Normal operational conditions reported.")
