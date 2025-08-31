"""
province_reference.py

Mapping utilities for Indonesian provinces using official BPS codes.

Includes:
- province_code_map: dict {province_name -> province_code}
- get_province_code(): fetch code by province name
- get_province_name_by_code(): fetch name by province code
"""

from __future__ import annotations

# === Official province code map (BPS standard) ===
province_code_map: dict[str, int] = {
    "Aceh": 11,
    "Sumatra Utara": 12,
    "Sumatra Barat": 13,
    "Riau": 14,
    "Jambi": 15,
    "Sumatera Selatan": 16,
    "Bengkulu": 17,
    "Lampung": 18,
    "Kepulauan Bangka Belitung": 19,
    "Kepulauan Riau": 21,
    "DKI Jakarta": 31,
    "Jawa Barat": 32,
    "Jawa Tengah": 33,
    "DI Yogyakarta": 34,
    "Jawa Timur": 35,
    "Banten": 36,
    "Bali": 51,
    "Nusa Tenggara Barat": 52,
    "Nusa Tenggara Timur": 53,
    "Kalimantan Barat": 61,
    "Kalimantan Tengah": 62,
    "Kalimantan Selatan": 63,
    "Kalimantan Timur": 64,
    "Kalimantan Utara": 65,
    "Sulawesi Utara": 71,
    "Sulawesi Tengah": 72,
    "Sulawesi Selatan": 73,
    "Sulawesi Tenggara": 74,
    "Gorontalo": 75,
    "Sulawesi Barat": 76,
    "Maluku": 81,
    "Maluku Utara": 82,
    "Papua": 91,
    "Papua Barat": 92,
}

# Reverse lookup (code -> name)
province_name_map: dict[int, str] = {v: k for k, v in province_code_map.items()}


def get_province_code(name: str) -> int | None:
    """
    Get official BPS province code from province name.

    Args:
        name (str): Province name (e.g., "Jawa Barat")

    Returns:
        int | None: Province code if found, else None
    """
    return province_code_map.get(name)


def get_province_name_by_code(code: int) -> str | None:
    """
    Get province name from official BPS code.

    Args:
        code (int): Province code (e.g., 32)

    Returns:
        str | None: Province name if found, else None
    """
    return province_name_map.get(code)
