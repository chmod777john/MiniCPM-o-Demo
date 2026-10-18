"""GCJ-02 / WGS-84 conversion and distance helpers.

AMap returns GCJ-02 ("Mars") coordinates. The training data stores WGS-84, so
every coordinate that leaves ``map_search`` is converted; coordinates sent to
AMap are converted the other way.
"""

from __future__ import annotations

import math
from typing import Tuple


_A = 6378245.0                      # Krasovsky 1940 semi-major axis
_EE = 0.00669342162296594323        # Krasovsky 1940 eccentricity squared
_EARTH_RADIUS_M = 6371000.0


def _out_of_china(latitude: float, longitude: float) -> bool:
    return not (0.8293 <= latitude <= 55.8271 and 72.004 <= longitude <= 137.8347)


def _transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _offset(latitude: float, longitude: float) -> Tuple[float, float]:
    d_lat = _transform_lat(longitude - 105.0, latitude - 35.0)
    d_lon = _transform_lon(longitude - 105.0, latitude - 35.0)
    rad_lat = latitude / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - _EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrt_magic) * math.pi)
    d_lon = (d_lon * 180.0) / (_A / sqrt_magic * math.cos(rad_lat) * math.pi)
    return d_lat, d_lon


def gcj02_to_wgs84(latitude: float, longitude: float) -> Tuple[float, float]:
    """Convert an AMap coordinate to WGS-84."""

    if _out_of_china(latitude, longitude):
        return latitude, longitude
    d_lat, d_lon = _offset(latitude, longitude)
    return latitude - d_lat, longitude - d_lon


def wgs84_to_gcj02(latitude: float, longitude: float) -> Tuple[float, float]:
    """Convert a WGS-84 coordinate to AMap's GCJ-02."""

    if _out_of_china(latitude, longitude):
        return latitude, longitude
    d_lat, d_lon = _offset(latitude, longitude)
    return latitude + d_lat, longitude + d_lon


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))
