import os

from run import needs_build


def test_the_dashboard_is_rebuilt_only_when_missing_or_older_than_its_sources(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "App.jsx"
    source.write_text("x")
    assert needs_build(tmp_path)

    built = tmp_path / "dist" / "index.html"
    built.parent.mkdir()
    built.write_text("x")
    os.utime(source, (1_000, 1_000))
    os.utime(built, (2_000, 2_000))
    assert not needs_build(tmp_path)

    os.utime(source, (3_000, 3_000))
    assert needs_build(tmp_path)
