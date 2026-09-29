import pytest

from backend.engine import news_ingestion
from backend.engine.news_ingestion import DynamicNewsIngestor


class Reply:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def current(code, wind=10.0, precipitation=0.0):
    return {"current": {"weather_code": code, "wind_speed_10m": wind, "precipitation": precipitation}}


@pytest.mark.parametrize("body, condition, description", [
    (current(0), "clear", "clear"),
    (current(3), "clear", "cloudy"),
    (current(45), "rainy", "fog"),
    (current(61, precipitation=1.2), "rainy", "rain"),
    (current(95), "stormy", "thunderstorm"),
    (current(2, wind=75.0), "stormy", "cloudy"),  # gale-force wind on a dry day
])
def test_open_meteo_readings_become_the_delay_models_condition(monkeypatch, body, condition, description):
    monkeypatch.setattr(news_ingestion.requests, "get", lambda *a, **kw: Reply(body))
    weather = DynamicNewsIngestor().fetch_weather(51.95, 4.14)
    assert weather["condition"] == condition and weather["description"] == description


def test_a_malformed_reply_means_no_weather_and_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(news_ingestion.requests, "get", lambda *a, **kw: calls.append(1) or Reply({"oops": 1}))
    ingestor = DynamicNewsIngestor()
    assert ingestor.fetch_weather(51.95, 4.14) is None
    assert ingestor.fetch_weather(51.95, 4.14) is None and len(calls) == 1  # retried later, not on every request


def test_weather_is_not_fetched_before_the_nlp_engine_is_ready(recommender, monkeypatch):
    def fail(lat, lon):
        raise AssertionError("fetched weather while live intel is off")
    assert not recommender.nlp.ready
    monkeypatch.setattr(recommender.news_ingestor, "fetch_weather", fail)
    assert recommender.recommend("Shanghai", "Rotterdam", live_intel=True)["live_intel"] == []


def feed(*entries):
    """An RSS body with (title, hours ago) items."""
    import email.utils
    import time
    items = "".join(f"<item><title>{t}</title><pubDate>{email.utils.formatdate(time.time() - h * 3600)}</pubDate></item>"
                    for t, h in entries)
    return f"<rss><channel>{items}</channel></rss>".encode()


def test_only_recent_headlines_count_newest_first(monkeypatch):
    class Page:
        content = feed(("Old strike at Rotterdam", 24 * 30), ("Storm hits Rotterdam", 5), ("Rotterdam queue grows", 1))

        def raise_for_status(self):
            pass
    monkeypatch.setattr(news_ingestion.requests, "get", lambda *a, **kw: Page())
    assert DynamicNewsIngestor().fetch_headlines("Rotterdam") == "Rotterdam queue grows | Storm hits Rotterdam"
