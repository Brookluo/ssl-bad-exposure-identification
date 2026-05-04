"""Tests for src.decam_info — CCD mappings and reason bitmask decoders."""
from __future__ import annotations

from src import decam_info


class TestCcdMappings:
    def test_ccdname2num_has_62_entries(self) -> None:
        assert len(decam_info.ccdname2num) == 62

    def test_ccdname2num_known_values(self) -> None:
        assert decam_info.ccdname2num["S1"] == 25
        assert decam_info.ccdname2num["S29"] == 1
        assert decam_info.ccdname2num["N31"] == 62

    def test_ccdnum2name_is_inverse(self) -> None:
        for name, num in decam_info.ccdname2num.items():
            assert decam_info.ccdnum2name[num] == name

    def test_ccdnum_li_m1_excludes_61(self) -> None:
        assert 61 not in decam_info.ccdnum_li_m1
        assert len(decam_info.ccdnum_li_m1) == 61

    def test_ccdnum_li_m2_excludes_61_and_2(self) -> None:
        assert 61 not in decam_info.ccdnum_li_m2
        assert 2 not in decam_info.ccdnum_li_m2
        assert len(decam_info.ccdnum_li_m2) == 60


class TestDecodeReason:
    def test_bit_0_yields_bad_wcscal(self) -> None:
        assert decam_info.decode_reason(1) == ["Bad_WCSCAL"]

    def test_bit_1_yields_saturated(self) -> None:
        assert decam_info.decode_reason(2) == ["Saturated"]

    def test_bit_0_and_1_yields_both(self) -> None:
        assert decam_info.decode_reason(3) == ["Bad_WCSCAL", "Saturated"]

    def test_zero_returns_empty_list(self) -> None:
        assert decam_info.decode_reason(0) == []

    def test_return_num_mode(self) -> None:
        assert decam_info.decode_reason(1, return_num=True) == [0]
        assert decam_info.decode_reason(3, return_num=True) == [0, 1]


class TestDecodeViSource:
    def test_bit_0_yields_empty(self) -> None:
        assert decam_info.decode_vi_source(0) == []

    def test_bit_1_yields_rongpu(self) -> None:
        assert decam_info.decode_vi_source(2) == ["Rongpu"]

    def test_bit_2_yields_alex(self) -> None:
        assert decam_info.decode_vi_source(4) == ["Alex"]

    def test_both_bits(self) -> None:
        assert decam_info.decode_vi_source(6) == ["Rongpu", "Alex"]


class TestDecodeMlLabel:
    def test_label_0_is_good(self) -> None:
        assert decam_info.decode_ml_label([0]) == ["good"]

    def test_label_1_is_bad_wcscal(self) -> None:
        assert decam_info.decode_ml_label([1]) == ["Bad_WCSCAL"]

    def test_multiple_labels(self) -> None:
        result = decam_info.decode_ml_label([0, 1, 3])
        assert result == ["good", "Bad_WCSCAL", "Bad_seeing"]


class TestIsMiss2Ccd:
    def test_pre_2014_returns_false(self) -> None:
        assert decam_info.is_miss_2ccd(56000.0) is False

    def test_2014_2017_returns_true(self) -> None:
        assert decam_info.is_miss_2ccd(57000.0) is True

    def test_post_2017_returns_false(self) -> None:
        assert decam_info.is_miss_2ccd(58000.0) is False
