"""Tests for decam_qa.info — CCD mappings, reason bitmasks, filter dicts."""
import pytest
from decam_qa.info import (
    ccdname2num, ccdnum2name,
    reason_num_dict, reason_li, reason_source_dict,
    decode_reason, decode_vi_source, decode_ml_label,
    filter_dict, is_miss_2ccd,
)


class TestCCDMappings:
    def test_ccdname2num_all_keys(self):
        expected_names = set([f"S{i}" for i in range(1, 32)] + [f"N{i}" for i in range(1, 32)])
        assert set(ccdname2num.keys()) == expected_names, "Missing or extra CCD names"
        vals = list(ccdname2num.values())
        assert len(set(vals)) == len(vals), "CCD numbers must be unique"
        assert min(vals) == 1
        assert max(vals) == 62

    def test_ccdnum2name_roundtrip(self):
        for name, num in ccdname2num.items():
            assert ccdnum2name[num] == name, f"Roundtrip failed: {name} -> {num}"
        for num, name in ccdnum2name.items():
            assert ccdname2num[name] == num, f"Roundtrip failed: {num} -> {name}"


class TestReasonDicts:
    def test_reason_num_dict_consistency(self):
        core_indices = set(range(15))
        for name, idx in reason_num_dict.items():
            if idx in core_indices:
                continue
            assert reason_num_dict.get(reason_li[idx]) == idx, \
                f"Alias {name} points to {idx} but canonical name mismatches"

    def test_decode_reason_zero(self):
        assert decode_reason(0) == []

    def test_decode_reason_single_bit(self):
        assert decode_reason(2**5) == ["Nonoptimal_exp"]

    def test_decode_reason_multi_bit(self):
        result = decode_reason(2**1 | 2**2)
        assert "Saturated" in result
        assert "Clouds_transparency" in result
        assert len(result) == 2

    def test_decode_reason_return_num(self):
        result = decode_reason(2**3, return_num=True)
        assert result == [3]

    def test_encode_decode_roundtrip(self):
        test_cases = [1, 4, 7, 15, 2**13 | 2**14, 2**0 | 2**2 | 2**6]
        for bitmask in test_cases:
            names = decode_reason(bitmask)
            reencoded = 0
            for name in names:
                reencoded |= 2 ** reason_num_dict[name]
            assert reencoded == bitmask, \
                f"Roundtrip failed: {bitmask} -> {names} -> {reencoded}"


class TestViSource:
    def test_decode_vi_source_single(self):
        assert decode_vi_source(1) == ["Rongpu"]

    def test_decode_vi_source_combined(self):
        result = decode_vi_source(3)
        assert "Rongpu" in result
        assert "Alex" in result

    def test_decode_vi_source_zero(self):
        assert decode_vi_source(0) == []


class TestMLLabel:
    def test_decode_ml_label_good(self):
        assert decode_ml_label([0]) == ["good"]

    def test_decode_ml_label_bad(self):
        result = decode_ml_label([2])
        assert result[0] == reason_li[1]

    def test_decode_ml_label_multiple(self):
        result = decode_ml_label([0, 3, 7])
        assert result == ["good", reason_li[2], reason_li[6]]


class TestFilterDict:
    def test_filter_dict_coverage(self):
        for band in ["g", "r", "i", "z", "Y"]:
            assert band in filter_dict
        assert all(1 <= v <= 5 for v in filter_dict.values())


class TestIsMiss2CCD:
    def test_before_2014(self):
        assert not is_miss_2ccd(56000.0)

    def test_during_2014_2016(self):
        assert is_miss_2ccd(57000.0)

    def test_after_2017(self):
        assert not is_miss_2ccd(58000.0)
