"""
province_reference.py

Indonesian Provincial Administrative Code Reference System.

This module provides comprehensive mapping utilities for Indonesian provincial
administrative divisions using the official Statistical Central Bureau (BPS)
standardized coding system. The reference system ensures consistent provincial
identification across different datasets and analytical workflows.

Key Components:
    - INDONESIAN_PROVINCE_CODES: Official BPS province code mapping dictionary
    - get_province_code(): Province name to code lookup function  
    - get_province_name(): Province code to name reverse lookup function

Standards Compliance:
    - Follows official BPS (Badan Pusat Statistik) provincial coding standards
    - Maintains compatibility with government statistical databases
    - Supports both forward and reverse lookups for data integration

Dependencies:
    - No external dependencies (pure Python implementation)

Author: Teuku Hafiez Ramadhan
License: Apache License 2.0
"""

from __future__ import annotations

# === Official Indonesian Province Code Registry (BPS Standard) ===
INDONESIAN_PROVINCE_CODES: dict[str, int] = {
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

# Generate reverse lookup mapping for bidirectional access
INDONESIAN_PROVINCE_NAMES: dict[int, str] = {code: name for name, code in INDONESIAN_PROVINCE_CODES.items()}


def get_province_code(province_name: str) -> int | None:
    """
    Retrieve the official BPS administrative code for a given Indonesian province name.
    
    This function provides a standardized lookup mechanism for converting province
    names to their corresponding official statistical codes, ensuring consistency
    across data integration and analysis workflows.
    
    Args:
        province_name (str): Full Indonesian province name (e.g., "Jawa Barat", "DKI Jakarta")
        
    Returns:
        int | None: Official BPS province code if the province name is found,
                   None if the province name is not recognized
                   
    Example:
        >>> code = get_province_code("Jawa Barat")
        >>> print(f"Province code for Jawa Barat: {code}")
        Province code for Jawa Barat: 32
        
    Note:
        Province names must match exactly with the official BPS naming conventions
        stored in the INDONESIAN_PROVINCE_CODES registry.
    """
    return INDONESIAN_PROVINCE_CODES.get(province_name)


def get_province_name(province_code: int) -> str | None:
    """
    Retrieve the official Indonesian province name from its BPS administrative code.
    
    This function provides reverse lookup functionality, converting numerical
    province codes back to their corresponding full province names for reporting
    and visualization purposes.
    
    Args:
        province_code (int): Official BPS province code (e.g., 32 for Jawa Barat)
        
    Returns:
        str | None: Full Indonesian province name if the code is valid,
                   None if the province code is not recognized
                   
    Example:
        >>> name = get_province_name(32)
        >>> print(f"Province name for code 32: {name}")
        Province name for code 32: Jawa Barat
        
    Note:
        This function serves as the inverse operation to get_province_code(),
        enabling bidirectional conversion between names and codes.
    """
    return INDONESIAN_PROVINCE_NAMES.get(province_code)
