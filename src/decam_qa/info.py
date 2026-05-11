"""DECam instrument information: CCD mappings, reason classification scheme, filter codes.

Provides static lookup tables for DECam's 62-CCD focal plane, a 15-category
bitmask-based bad-exposure reason scheme, and utility functions for decoding
bitmasks into human-readable reason strings.
"""
from typing import Iterable, List


ccdname2num = {
    "S1": 25, "S2": 26, "S3": 27, "S4": 28, "S5": 29,
    "S6": 30, "S7": 31, "S8": 19, "S9": 20, "S10": 21,
    "S11": 22, "S12": 23, "S13": 24, "S14": 13, "S15": 14,
    "S16": 15, "S17": 16, "S18": 17, "S19": 18, "S20": 8,
    "S21": 9, "S22": 10, "S23": 11, "S24": 12, "S25": 4,
    "S26": 5, "S27": 6, "S28": 7, "S29": 1, "S30": 2, "S31": 3,
    "N1": 32, "N2": 33, "N3": 34, "N4": 35, "N5": 36,
    "N6": 37, "N7": 38, "N8": 39, "N9": 40, "N10": 41,
    "N11": 42, "N12": 43, "N13": 44, "N14": 45, "N15": 46,
    "N16": 47, "N17": 48, "N18": 49, "N19": 50, "N20": 51,
    "N21": 52, "N22": 53, "N23": 54, "N24": 55, "N25": 56,
    "N26": 57, "N27": 58, "N28": 59, "N29": 60, "N30": 61, "N31": 62,
}
ccdnum2name = {v: k for k, v in ccdname2num.items()}

_li = list(range(1, 63))
_li.remove(61)
ccdnum_li_m1 = tuple(_li)

_li2 = list(range(1, 63))
_li2.remove(61)
_li2.remove(2)
ccdnum_li_m2 = tuple(_li2)


def is_miss_2ccd(mjd_obs: float) -> bool:
    """Check whether CCD S30 (number 2) was non-functional at this observation date.

    S30/CCD 2 was non-functional between 2014 and 2017.
    """
    from astropy.time import Time
    byear = Time(mjd_obs, format="mjd", scale="utc").byear
    return 2014 <= byear < 2017


reason_num_dict = {
    "Bad_WCSCAL": 0, "Saturated": 1, "Clouds_transparency": 2,
    "Bad_seeing": 3, "PSF": 4, "Nonoptimal_exp": 5,
    "Ghost_scatter": 6, "NObjects": 7, "Bad_CCD": 8,
    "Noise": 9, "Fringing": 10, "Canopus": 11,
    "Wonky": 12, "Telescope_moving": 13, "Out_of_focus": 14,
    "Clouds": 2, "Ghosting": 6, "Telescope_tracking": 13, "Readout": 9,
}

reason_li = (
    "Bad_WCSCAL", "Saturated", "Clouds_transparency", "Bad_seeing",
    "PSF", "Nonoptimal_exp", "Ghost_scatter", "NObjects",
    "Bad_CCD", "Noise", "Fringing", "Canopus", "Wonky",
    "Telescope_moving", "Out_of_focus",
)

reason_source_dict = {"Rongpu": 1, "Alex": 2}
filter_dict = {"g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}


def decode_reason(bit_reason: int, return_num: bool = False) -> List:
    """Decode a reason bitmask into a list of reason strings (or integer indices).

    Parameters
    ----------
    bit_reason : int
        Bitmask encoding bad-exposure reasons. Bit i corresponds to reason_li[i].
    return_num : bool, optional
        If True, return integer indices instead of reason name strings.

    Returns
    -------
    List
        List of reason strings (or ints if return_num=True).
    """
    if return_num:
        return [i for i in range(len(reason_li)) if 2**i & bit_reason]
    return [reason_li[i] for i in range(len(reason_li)) if 2**i & bit_reason]


def decode_vi_source(bit_source: int) -> List[str]:
    """Decode a visual-inspection source bitmask into source names.

    Parameters
    ----------
    bit_source : int
        Bitmask. Bit 0 = Rongpu, Bit 1 = Alex.

    Returns
    -------
    List[str]
        Source names that flagged this exposure.
    """
    return [name for name, i in reason_source_dict.items() if i & bit_source]


def decode_ml_label(ml_label: Iterable) -> List[str]:
    """Convert ML model label integers to reason strings.

    Parameters
    ----------
    ml_label : Iterable
        Integer labels where 0 = good, >=1 = reason_li[label-1].

    Returns
    -------
    List[str]
        Human-readable reason strings.
    """
    return [reason_li[l - 1] if l > 0 else "good" for l in ml_label]
