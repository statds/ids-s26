"""
district_zip_convert.py

Utility to standardize / map Connecticut school district names, optionally using ZIP codes.

This is a Python rewrite of 0_district_zip_convert.R used in the course case study.

Usage
-----
from district_zip_convert import district_zip_convert

# Names only:
df = district_zip_convert(school_district_series)

# Names + zip:
df = district_zip_convert(school_district_series, zip_series)
"""

from __future__ import annotations

from typing import Iterable, Optional, Union
import pandas as pd

# --- Name-based recodes (District_Name -> school_district) ---
NAME_MAP = {
    # Regional School District 01 (Canaan/Cornwall/Kent/North Canaan/Salisbury/Sharon)
    "Canaan School District": "Regional School District 01",
    "Cornwall School District": "Regional School District 01",
    "Kent School District": "Regional School District 01",
    "North Canaan School District": "Regional School District 01",
    "Salisbury School District": "Regional School District 01",
    "Sharon School District": "Regional School District 01",

    # Brookfield Area School District
    "Sherman School District": "Brookfield Area School District",
    "New Milford School District": "Brookfield Area School District",
    "New Fairfield School District": "Brookfield Area School District",
    "Brookfield School District": "Brookfield Area School District",

    # Region 05 (Woodbridge)
    "Woodbridge School District": "Regional School District 05",

    # Salem, East Lyme
    "Salem School District": "Salem, East Lyme",
    "East Lyme School District": "Salem, East Lyme",

    # Norwich School District
    "Bozrah School District": "Norwich School District",
    "Franklin School District": "Norwich School District",
    "Sprague School District": "Norwich School District",
    "Preston School District": "Norwich School District",
    # R note: this is the correct label to recode:
    "Norwich Free Academy": "Norwich School District",

    # Griswold School District
    "Lisbon School District": "Griswold School District",
    "Voluntown School District": "Griswold School District",

    # Plainfield School District
    "Sterling School District": "Plainfield School District",

    # Woodstock School District
    "Canterbury School District": "Woodstock School District",
    "Brooklyn School District": "Woodstock School District",
    "Pomfret School District": "Woodstock School District",
    "Eastford School District": "Woodstock School District",
    "Union School District": "Woodstock School District",
    "Woodstock Academy District": "Woodstock School District",

    # Regional School District 11
    "Scotland School District": "Regional School District 11",

    # Regional School District 06
    "Goshen School District": "Regional School District 06",

    # Regional School District 07
    "Barkhamsted School District": "Regional School District 07",
    "Colebrook School District": "Regional School District 07",
    "New Hartford School District": "Regional School District 07",
    "Norfolk School District": "Regional School District 07",
    "Winchester School District": "Regional School District 07",

    # Granby School District
    "Hartland School District": "Granby School District",

    # Regional School District 19
    "Willington School District": "Regional School District 19",

    # Bolton School District
    "Columbia School District": "Bolton School District",
}

# --- ZIP-based overrides (zip -> school_district) ---
ZIP_MAP = {
    # remove 6254 from Plainfield School District
    6254: "Norwich School District",

    6001: "Avon School District",
    6002: "Bloomfield School District",
    6062: "Plainville School District",
    6076: "Stafford School District",
    6078: "Suffield School District",
    6095: "Windsor School District",
    6098: "Regional School District 07",
    6105: "Hartford School District",
    6110: "West Hartford School District",
    6111: "Newington School District",
    6117: "West Harford School District",  # as in R file (possible typo)
    6118: "East Hartford School District",
    6120: "Hartford School District",
    6239: "Killingly School District",
    6320: "New London School District",
    6333: "Salem, East Lyme",
    6355: "Groton School District",
    6359: "North Stonington School District",
    6371: "Regional School District 18",
    6410: "Cheshire School District",
    6437: "Guilford School District",
    6441: "Regional School District 17",
    6457: "Middletown School District",
    6461: "Milford School District",
    6468: "Monroe School District",
    6471: "North Branford School District",
    6473: "North Haven School District",
    6489: "Southington School District",
    6492: "Wallingford School District",
    6510: "New Haven School District",
    6511: "New Haven School District",
    6512: "East Haven School District",
    6513: "New Haven School District",
    6514: "Hamden School District",
    6515: "New Haven School District",
    6516: "West Haven School District",
    6517: "Hamden School District",
    6604: "Bridgeport School District",
    6606: "Bridgeport School District",
    6608: "Bridgeport School District",
    6610: "Bridgeport School District",
    6759: "Litchfield School District",
    6770: "Naugatuck School District",
    6790: "Torrington School District",
    6901: "Stamford School District",
    6902: "Stamford School District",
}

def district_zip_convert(
    school_district: Union[Iterable[str], pd.Series],
    zip: Optional[Union[Iterable[int], pd.Series]] = None,
) -> pd.DataFrame:
    """
    Standardize/massage school district names, optionally using ZIP-based overrides.

    Parameters
    ----------
    school_district:
        Iterable/Series of district names.
    zip:
        Optional iterable/Series of ZIP codes (integers). If provided, ZIP-based rules
        are applied after name-based rules.

    Returns
    -------
    pd.DataFrame with:
      - school_district (category dtype)
      - zip (if provided)
    """
    sd = pd.Series(list(school_district), dtype="string").copy()

    if zip is not None:
        z = pd.Series(list(zip))
    else:
        z = None

    # Match the R function's working frame
    sdl = pd.DataFrame({"District_Name": sd, "school_district": sd})
    if z is not None:
        sdl["zip"] = z

    # Name-based recodes
    sdl["school_district"] = sdl["school_district"].map(lambda x: NAME_MAP.get(str(x), str(x)))

    # ZIP-based overrides (applied last)
    if z is not None:
        sdl["zip"] = pd.to_numeric(sdl["zip"], errors="coerce")

        def zip_override(row):
            zz = row["zip"]
            if pd.isna(zz):
                return row["school_district"]
            try:
                z_int = int(zz)
            except Exception:
                return row["school_district"]
            return ZIP_MAP.get(z_int, row["school_district"])

        sdl["school_district"] = sdl.apply(zip_override, axis=1)

    # R factor -> pandas categorical
    sdl["school_district"] = sdl["school_district"].astype("category")

    if z is not None:
        return sdl[["school_district", "zip"]]
    return sdl[["school_district"]]
