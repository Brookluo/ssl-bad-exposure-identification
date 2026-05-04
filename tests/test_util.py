"""Tests for src.util — HTML parsing and webpage generation."""
from __future__ import annotations

from pathlib import Path
from src.util import get_info_from_html, make_webpage


class TestGetInfoFromHtml:
    def test_returns_list_of_entries(self, sample_html_path: str) -> None:
        entries = get_info_from_html(sample_html_path)
        assert isinstance(entries, list)
        assert len(entries) == 2

    def test_entry_format(self, sample_html_path: str) -> None:
        entries = get_info_from_html(sample_html_path)
        first = entries[0]
        assert "c4d_200101_000000_ooi_g_ls9.fits.fz" in first[0]
        assert "123456" in first[1]
        assert first[-1].startswith('<td><img src=".')

    def test_empty_html_returns_empty_list(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.html"
        empty.write_text("")
        entries = get_info_from_html(str(empty))
        assert entries == []


class TestMakeWebpage:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        master = [["a", "1", "x"], ["b", "2", "y"]]
        make_webpage(master, [0, 1], tmp_path, "test", num_element=1)
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1

    def test_html_contains_table(self, tmp_path: Path) -> None:
        master = [["a", "1", "x"]]
        make_webpage(master, [0], tmp_path, "single", num_element=10)
        html = (tmp_path / "0_single_1_1.html").read_text()
        assert "<table>" in html
        assert "</table>" in html
