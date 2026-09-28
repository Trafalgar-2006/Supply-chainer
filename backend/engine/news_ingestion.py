import feedparser
import requests
import urllib.parse
import time
from typing import Optional

FEED_TIMEOUT_S = 2.0

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

    def fetch_headlines(self, location: str, max_items: int = 3) -> Optional[str]:
        """Latest logistics-disruption headlines for a place, or None if the feed is unavailable.

        None means "no live signal", which callers must not confuse with the static
        fallback reports.
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
            entries = feedparser.parse(response.content).entries
            if entries:
                content = " | ".join(entry.title for entry in entries[:max_items])
        except Exception as e:
            print(f"[NEWS] Feed unavailable for {location!r}: {e}")
        self.cache[key] = (now, content)
        return content

    def get_latest_news(self, location: str, transport_mode: str) -> str:
        """Live headlines for a location, or the mode's static fallback report."""
        return self.fetch_headlines(location) or self.fallback_news.get(
            transport_mode.lower(), "Normal operational conditions reported.")
