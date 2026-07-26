"""Map profile_meta SIC-style labels → coarse sector buckets for Satellite 1.

Local ``profile_meta['sector']`` holds SIC industry descriptions (not GICS).
This module assigns each description to one of:
  Technology, Healthcare, Consumer Staples, Industrials, Financials, Energy,
  Other, Unknown.

No new data source — keyword rules over existing metadata only.
"""

from __future__ import annotations

from typing import Dict, Optional

# Target satellite buckets (Task 2)
SATELLITE_SECTORS = (
    "Healthcare",
    "Consumer Staples",
    "Industrials",
    "Financials",
    "Energy",
)

# Ordered rules: first match wins (more specific before broad).
_RULES = [
    (
        "Healthcare",
        (
            "PHARMACEUTICAL",
            "BIOLOGICAL PRODUCTS",
            "SURGICAL",
            "MEDICAL",
            "ELECTROMEDICAL",
            "ORTHOPEDIC",
            "HEALTH",
            "HOSPITAL",
            "DIAGNOSTIC",
            "IN VITRO",
            "MEDICINAL",
            "OPHTHALMIC",
            "DRUG STORES",
            "DRUGGISTS",
        ),
    ),
    (
        "Energy",
        (
            "CRUDE PETROLEUM",
            "NATURAL GAS",
            "OIL & GAS",
            "OIL AND GAS",
            "PETROLEUM",
            "PIPE LINE",  # rare
        ),
    ),
    (
        "Financials",
        (
            "BANK",
            "SAVINGS INSTITUTION",
            "FINANCE",
            "CREDIT INSTITUTION",
            "LOAN BROKER",
            "INVESTMENT ADVICE",
            "INSURANCE",
            "REAL ESTATE INVESTMENT TRUST",
            "SECURITY & COMMODITY",
            "COMMODITY CONTRACTS",
            "BLANK CHECKS",  # SPACs — financial vehicles
        ),
    ),
    (
        "Consumer Staples",
        (
            "FOOD",
            "BEVERAGE",
            "SUGAR",
            "CONFECTIONERY",
            "GROCERY",
            "CANNED",
            "PERFUMES",
            "COSMETICS",
            "TOILET PREPARATIONS",
            "DRUG STORES",  # also healthcare — already matched above if drug stores
        ),
    ),
    (
        "Technology",
        (
            "SOFTWARE",
            "SEMICONDUCTOR",
            "COMPUTER",
            "ELECTRONIC COMPUTER",
            "COMPUTER PERIPHERAL",
            "COMPUTER STORAGE",
            "COMPUTER COMMUNICATIONS",
            "RADIO & TV BROADCASTING & COMMUNICATIONS EQUIPMENT",
            "COMMUNICATIONS EQUIPMENT",
            "TELEPHONE & TELEGRAPH APPARATUS",
            "SERVICES-COMPUTER",
            "SERVICES-PREPACKAGED",
            "DATA PROCESSING",
        ),
    ),
    (
        "Industrials",
        (
            "INDUSTRIAL",
            "MACHINERY",
            "ENGINES & TURBINES",
            "MOTORS & GENERATORS",
            "AIRCRAFT",
            "ORDNANCE",
            "TRUCKING",
            "AIR TRANSPORTATION",
            "WATER TRANSPORTATION",
            "ARRANGEMENT OF TRANSPORTATION",
            "FREIGHT",
            "CONSTRUCTION",
            "SPECIAL TRADE CONTRACTORS",
            "FABRICATED",
            "STEEL WORKS",
            "METAL FORGINGS",
            "MEASURING & CONTROLLING",
            "INDUSTRIAL INSTRUMENTS",
            "SEARCH, DETECTION",
            "AIR-COND",
            "ELECTRICAL MACHINERY",
            "ELECTRIC LIGHTING",
            "CUTLERY",
            "HARDWARE",
            "REFUSE",
            "HAZARDOUS WASTE",
            "EQUIPMENT RENTAL",
            "HELP SUPPLY",
            "TO DWELLINGS",
            "BUSINESS SERVICES",
            "MANAGEMENT CONSULTING",
            "ADVERTISING",
        ),
    ),
]


def normalize_sector_label(raw: Optional[str]) -> str:
    if raw is None:
        return "Unknown"
    s = str(raw).strip().upper()
    if not s or s == "UNKNOWN":
        return "Unknown"
    for bucket, keys in _RULES:
        for k in keys:
            if k in s:
                return bucket
    return "Other"


def sector_of_meta(meta: Optional[dict]) -> str:
    if not meta:
        return "Unknown"
    # Prefer 'sector' (SIC desc in this dataset); fall back to industry.
    raw = meta.get("sector") or meta.get("industry") or meta.get("Industry")
    return normalize_sector_label(raw)


def build_symbol_sector_map(profile_meta: Dict[str, dict]) -> Dict[str, str]:
    return {str(sym): sector_of_meta(meta) for sym, meta in (profile_meta or {}).items()}
