import pytest

from backend.engine.threat_intelligence import CARFFilter

carf = CARFFilter()

SEA_NEWS = "Vessel grounding blocks canal traffic; ships queue at the port."
AIR_NEWS = "Airport closed and all cargo flights grounded after a system outage."
RAIL_NEWS = "Locomotive derailment shuts the main rail line and freight station."
ROAD_NEWS = "Truck convoy stranded after highway bridge collapse."
NEUTRAL_NEWS = "Severe cyclone makes landfall; authorities declare a regional emergency."


@pytest.mark.parametrize("mode, news", [
    ("sea", SEA_NEWS),
    ("air", AIR_NEWS),
    ("rail", RAIL_NEWS),
    ("road", ROAD_NEWS),
])
def test_threat_about_the_same_mode_is_kept(mode, news):
    assert carf.apply_filter(0.8, news, mode) == 0.8


@pytest.mark.parametrize("mode, news", [
    ("sea", AIR_NEWS),
    ("air", SEA_NEWS),
    ("rail", ROAD_NEWS),
    ("road", RAIL_NEWS),
    ("sea", RAIL_NEWS),
    ("air", ROAD_NEWS),
])
def test_threat_about_a_different_mode_is_dropped(mode, news):
    assert carf.apply_filter(0.8, news, mode) == 0.0


@pytest.mark.parametrize("mode", ["sea", "air", "rail", "road"])
def test_mode_neutral_threat_is_kept_for_every_mode(mode):
    assert carf.apply_filter(0.6, NEUTRAL_NEWS, mode) == 0.6


@pytest.mark.parametrize("news", [
    "Typhoon warning issued by the weather station; heavy rain expected.",
    "Forecasters track the storm as it heads for the coast.",
    "Home delivery demand surges ahead of the holidays.",
])
@pytest.mark.parametrize("mode", ["sea", "air", "rail", "road"])
def test_generic_words_do_not_misfile_mode_neutral_news(news, mode):
    assert carf.apply_filter(0.6, news, mode) == 0.6


def test_ies_plurals_are_recognised():
    assert carf.modes_mentioned("Lorries stranded at the border") == {"road"}


def test_news_about_both_modes_is_kept():
    news = "Storm closes the airport and the container port."
    assert carf.apply_filter(0.7, news, "sea") == 0.7
    assert carf.apply_filter(0.7, news, "air") == 0.7


def test_airport_is_not_mistaken_for_a_seaport():
    assert carf.apply_filter(0.7, "Airport strike halts operations.", "sea") == 0.0


def test_keywords_match_through_punctuation_and_plurals():
    assert carf.apply_filter(0.5, "Ports, docks and vessels idle.", "air") == 0.0
    assert carf.apply_filter(0.5, "Ports, docks and vessels idle.", "sea") == 0.5


def test_non_positive_scores_stay_zero():
    assert carf.apply_filter(0.0, SEA_NEWS, "sea") == 0.0
    assert carf.apply_filter(-0.2, SEA_NEWS, "sea") == 0.0
