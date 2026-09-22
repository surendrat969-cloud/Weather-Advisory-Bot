"""
Weather data fetching from Open-Meteo API.
Geocoding city names and fetching current conditions.
No API key required.
"""
import httpx
from typing import Optional
from dataclasses import dataclass

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_FIELDS = [
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weather_code",
    "relative_humidity_2m",
    "apparent_temperature",
    "wind_gusts_10m",
]


@dataclass
class WeatherData:
    location_name: str
    latitude: float
    longitude: float
    time: str
    temperature_2m: Optional[float]
    wind_speed_10m: Optional[float]
    precipitation: Optional[float]
    precipitation_probability: Optional[float]
    uv_index: Optional[float]
    weather_code: Optional[int]
    relative_humidity_2m: Optional[float]
    apparent_temperature: Optional[float]
    wind_gusts_10m: Optional[float]
    raw: dict  # full API response for auditability


def geocode_city(city: str) -> tuple[float, float, str]:
    """
    Resolve city name to (lat, lon, display_name).
    Raises ValueError if city not found.
    Raises httpx.HTTPError on network failure.
    """
    resp = httpx.get(
        GEOCODING_URL,
        params={"name": city, "count": 5, "language": "en"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results")
    if not results:
        raise ValueError(
            f"Location '{city}' could not be resolved. No geocoding results returned."
        )
    # Pick first result
    r = results[0]
    display = (
        f"{r.get('name', city)}, {r.get('admin1', '')}, {r.get('country', '')}".strip(
            ", "
        )
    )
    return r["latitude"], r["longitude"], display


def fetch_weather(lat: float, lon: float, location_name: str) -> WeatherData:
    """
    Fetch current weather for given coordinates.
    Raises httpx.HTTPError on network failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(CURRENT_FIELDS),
        "timezone": "auto",
    }
    resp = httpx.get(FORECAST_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})

    return WeatherData(
        location_name=location_name,
        latitude=lat,
        longitude=lon,
        time=current.get("time", "unknown"),
        temperature_2m=current.get("temperature_2m"),
        wind_speed_10m=current.get("wind_speed_10m"),
        precipitation=current.get("precipitation"),
        precipitation_probability=current.get("precipitation_probability"),
        uv_index=current.get("uv_index"),
        weather_code=current.get("weather_code"),
        relative_humidity_2m=current.get("relative_humidity_2m"),
        apparent_temperature=current.get("apparent_temperature"),
        wind_gusts_10m=current.get("wind_gusts_10m"),
        raw=data,
    )


def get_weather_for_city(city: str) -> WeatherData:
    """High-level: geocode + fetch. Raises ValueError or httpx.HTTPError."""
    lat, lon, display = geocode_city(city)
    return fetch_weather(lat, lon, display)
