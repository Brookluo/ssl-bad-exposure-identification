# Provide info for DECam exposures
from typing import Iterable, List


ccdname2num = {
    "S1": 25,
    "S2": 26,
    "S3": 27,
    "S4": 28,
    "S5": 29,
    "S6": 30,
    "S7": 31,
    "S8": 19,
    "S9": 20,
    "S10": 21,
    "S11": 22,
    "S12": 23,
    "S13": 24,
    "S14": 13,
    "S15": 14,
    "S16": 15,
    "S17": 16,
    "S18": 17,
    "S19": 18,
    "S20": 8,
    "S21": 9,
    "S22": 10,
    "S23": 11,
    "S24": 12,
    "S25": 4,
    "S26": 5,
    "S27": 6,
    "S28": 7,
    "S29": 1,
    "S30": 2,
    "S31": 3,
    "N1": 32,
    "N2": 33,
    "N3": 34,
    "N4": 35,
    "N5": 36,
    "N6": 37,
    "N7": 38,
    "N8": 39,
    "N9": 40,
    "N10": 41,
    "N11": 42,
    "N12": 43,
    "N13": 44,
    "N14": 45,
    "N15": 46,
    "N16": 47,
    "N17": 48,
    "N18": 49,
    "N19": 50,
    "N20": 51,
    "N21": 52,
    "N22": 53,
    "N23": 54,
    "N24": 55,
    "N25": 56,
    "N26": 57,
    "N27": 58,
    "N28": 59,
    "N29": 60,
    "N30": 61,
    "N31": 62,
}
ccdnum2name = {v: k for k, v in ccdname2num.items()}

ccdnum_li_m1 = list(range(1, 63))
ccdnum_li_m1.remove(61)  # missing 61
ccdnum_li_m1 = tuple(ccdnum_li_m1)  # immutable

ccdnum_li_m2 = list(range(1, 63))
ccdnum_li_m2.remove(61)  # missing 61
ccdnum_li_m2.remove(2)  # missing 2
ccdnum_li_m2 = tuple(ccdnum_li_m2)


def is_miss_2ccd(mjd_obs: float):
    from astropy.time import Time

    # matched = tab[tab["expnum"] == expnum]
    # assert len(matched) > 0, f"expnum {expnum} not found in the table"
    return (
        True
        if (
            time := Time(
                mjd_obs,
                format="mjd",
                scale="utc",
            ).byear
        )
        >= 2014
        and time < 2017
        else False
    )


# Reasons for bad exposures
reason_num_dict = {
    # "good": 0
    "Bad_WCSCAL": 0,
    "Saturated": 1,
    "Clouds_transparency": 2,
    "Bad_seeing": 3,
    "PSF": 4,
    "Nonoptimal_exp": 5,
    "Ghost_Scatter": 6,
    "NObjects": 7,
    "Bad_CCD": 8,
    "Noise": 9,
    "Fringing": 10,
    "Canopus": 11,
    "Wonky": 12,
    "Telescope_Moving": 13,
    "Out_of_focus": 14,
    "Clouds": 2,
    "Ghosting": 6,
    "Telescope_Tracking": 13,
    "Readout": 9,  # original num: 15
}
# removed the redaundant options
reason_li = (
    # "good"
    "Bad_WCSCAL",
    "Saturated",
    "Clouds_transparency",
    "Bad_seeing",
    "PSF",
    "Nonoptimal_exp",
    "Ghost_Scatter",
    "NObjects",
    "Bad_CCD",
    "Noise",
    "Fringing",
    "Canopus",
    "Wonky",
    "Telescope_Moving",
    "Out_of_focus",
    # 'Readout'
)
reason_source_dict = {"Rongpu": 1, "Alex": 2}
filter_dict = {"g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}


def decode_reason(bit_reason: int, return_num=False):
    """Decode the bitmask for issues with this exposure

    Params:
    --------
    bit_reason: int
        the bitmask reason for this image

    Returns:
    --------
    List
        a list of reasons causing it to be bad image

    """
    reason_list = reason_li
    if return_num:
        return [i for i in range(len(reason_list)) if 2**i & bit_reason]
    return [reason_list[i] for i in range(len(reason_list)) if 2**i & bit_reason]


def decode_vi_source(bit_source):
    # if reason_source_dict is None:
    # reason_source_dict = reason_source
    return [name for name, i in reason_source_dict.items() if 2**i & bit_source]


def decode_ml_label(ml_label: Iterable) -> List:
    """Decode the machine learning model generated labels

    Params:
    --------
    ml_label: Iterable
        the iterable array of ml labels

    Returns:
    --------
    List
        a list of decoded reason for each label

    """
    return [reason_li[l - 1] if l > 0 else "good" for l in ml_label]
