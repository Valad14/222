from __future__ import annotations

import colorsys
import hashlib
import json
from typing import Iterable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_REGION_BOUNDARIES = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Удмуртская Республика"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [51.05, 55.85],
                        [52.20, 55.72],
                        [53.85, 55.82],
                        [54.65, 56.35],
                        [54.72, 57.35],
                        [53.95, 58.25],
                        [52.30, 58.45],
                        [51.15, 57.95],
                        [50.75, 56.90],
                        [51.05, 55.85],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "Кировская область"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [46.00, 56.00],
                        [50.75, 56.00],
                        [51.15, 57.95],
                        [52.30, 58.45],
                        [51.20, 59.55],
                        [48.30, 59.75],
                        [46.00, 58.40],
                        [46.00, 56.00],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "Пермский край"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [53.95, 58.25],
                        [54.72, 57.35],
                        [55.85, 56.95],
                        [58.35, 57.60],
                        [59.20, 59.65],
                        [57.15, 60.70],
                        [54.65, 60.10],
                        [53.95, 58.25],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "Республика Татарстан"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [47.30, 53.95],
                        [50.20, 53.75],
                        [53.85, 55.82],
                        [52.20, 55.72],
                        [51.05, 55.85],
                        [49.10, 56.20],
                        [47.30, 55.40],
                        [47.30, 53.95],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "Республика Башкортостан"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [53.20, 51.55],
                        [56.85, 51.35],
                        [58.35, 54.45],
                        [55.85, 56.95],
                        [54.65, 56.35],
                        [53.85, 55.82],
                        [52.75, 54.30],
                        [53.20, 51.55],
                    ]
                ],
            },
        },
    ],
}

LOCAL_COORDINATES: dict[str, tuple[float, float]] = {
    "алнаши": (56.1873, 52.4794),
    "балезино": (57.9796, 53.0138),
    "вавож": (56.7752, 51.9300),
    "грахово": (56.0505, 51.9675),
    "завьялово": (56.7869, 53.3719),
    "игра": (57.5549, 53.0544),
    "каракулино": (56.0124, 53.7069),
    "кез": (57.8970, 53.7132),
    "кизнер": (56.2744, 51.5082),
    "киясьово": (56.3463, 53.1241),
    "киясово": (56.3463, 53.1241),
    "красногорское": (57.7047, 52.5008),
    "малая пурга": (56.5561, 52.9945),
    "селты": (57.3143, 52.1343),
    "сюмси": (57.1110, 51.6152),
    "ува": (56.9907, 52.1850),
    "шаркан": (57.2989, 53.8717),
    "якшур-бодья": (57.1834, 53.1562),
    "яр": (58.2457, 52.1058),
    "пугачёво": (56.5451, 52.7264),
    "пугачево": (56.5451, 52.7264),
    "новый": (56.7508, 53.0622),
}


def label_color(label: object, alpha: int = 190) -> list[int]:
    text = str(label or "нет данных").encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    hue = int(digest[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hls_to_rgb(hue, 0.50, 0.58)
    return [int(r * 255), int(g * 255), int(b * 255), alpha]


def label_color_hex(label: object) -> str:
    r, g, b, _ = label_color(label)
    return f"#{r:02x}{g:02x}{b:02x}"


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def convex_hull(points: Iterable[tuple[float, float]]) -> list[list[float]]:
    pts = sorted(set(points))
    if len(pts) < 3:
        return []
    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return []
    closed = hull + [hull[0]]
    return [[float(lon), float(lat)] for lon, lat in closed]


def is_city_type(value: object) -> bool:
    text = str(value or "").strip().lower().replace("ё", "е")
    return text in {"город", "г", "г."} or text.startswith("город ")


def without_city_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "settlement_type" not in df.columns:
        return df.copy()
    mask = ~df["settlement_type"].apply(is_city_type)
    return df[mask].copy()


def map_view_state(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["latitude", "longitude"])
    if valid.empty:
        return {"latitude": 57.1, "longitude": 53.2, "zoom": 6.5, "pitch": 0}
    lat_min, lat_max = valid["latitude"].min(), valid["latitude"].max()
    lon_min, lon_max = valid["longitude"].min(), valid["longitude"].max()
    lat_span = max(lat_max - lat_min, 0.2)
    lon_span = max(lon_max - lon_min, 0.2)
    span = max(lat_span, lon_span)
    if span > 5:
        zoom = 5.5
    elif span > 2.5:
        zoom = 6.2
    elif span > 1:
        zoom = 7.2
    else:
        zoom = 8.4
    return {
        "latitude": float((lat_min + lat_max) / 2),
        "longitude": float((lon_min + lon_max) / 2),
        "zoom": zoom,
        "pitch": 0,
    }


def build_areals(exploded_df: pd.DataFrame, group_col: str = "linguistic_unit") -> list[dict]:
    valid = without_city_points(exploded_df).dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty or group_col not in valid.columns:
        return []
    areals: list[dict] = []
    for label, group in valid.groupby(group_col):
        points = list(zip(group["longitude"].astype(float), group["latitude"].astype(float)))
        polygon = convex_hull(points)
        if len(polygon) < 4:
            continue
        color = label_color(label, alpha=28)
        areals.append(
            {
                "label": str(label),
                "polygon": polygon,
                "path": polygon,
                "count": int(group["settlement"].nunique()),
                "fill_color": color,
                "line_color": [color[0], color[1], color[2], 160],
            }
        )
    return sorted(areals, key=lambda item: (-item["count"], item["label"]))


def _unique_join(series: pd.Series, limit: int = 9) -> str:
    values: list[str] = []
    for value in series.dropna().astype(str):
        for item in value.split(";"):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f"; +{len(values) - limit}"
    return "; ".join(shown) + suffix


def aggregate_points(df: pd.DataFrame) -> pd.DataFrame:
    valid = without_city_points(df).dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = (
        valid.groupby(["region", "district", "settlement", "latitude", "longitude"], dropna=False)
        .agg(
            settlement_type=("settlement_type", "first"),
            landscape=("landscape", "first"),
            atlas_system=("atlas_system", _unique_join),
            question_type=("question_type", _unique_join),
            question_label=("question", _unique_join),
            unit_display=("unit_display", _unique_join),
            comments=("comment", _unique_join),
            record_count=("question", "size"),
            question_count=("question", "nunique"),
        )
        .reset_index()
    )
    grouped["tooltip"] = grouped.apply(
        lambda row: (
            f"<b>{row['settlement']}</b><br>"
            f"{row['district']}, {row['region']}<br>"
            f"Тип: {row['settlement_type']}<br>"
            f"Ландшафт: {row['landscape']}<br>"
            f"Вопросы: {row['question_label']}<br>"
            f"Единицы: {row['unit_display']}<br>"
            f"Комментарий: {row['comments'] or '—'}"
        ),
        axis=1,
    )
    return grouped


def add_point_visuals(points_df: pd.DataFrame, color_mode: str) -> pd.DataFrame:
    points_df = points_df.copy()
    if color_mode == "Ландшафт":
        points_df["color_label"] = points_df["landscape"].replace("", "не указан")
    elif color_mode == "Тип вопроса":
        points_df["color_label"] = points_df["question_type"].replace("", "не указан")
    elif color_mode == "Атлас":
        points_df["color_label"] = points_df["atlas_system"].replace("", "не указан")
    else:
        points_df["color_label"] = points_df["unit_display"].replace("", "нет данных")
    points_df["color"] = points_df["color_label"].apply(lambda value: label_color(value, alpha=210))
    points_df["outline_color"] = points_df["color_label"].apply(lambda value: label_color(value, alpha=255))
    points_df["radius_m"] = 5200 + points_df["record_count"].clip(0, 10).astype(int) * 450
    points_df["short_label"] = points_df["settlement"].astype(str).str.slice(0, 18)
    return points_df


def geocode_settlement(settlement: str, district: str = "", region: str = "") -> tuple[float, float, str] | None:
    """Find coordinates for a settlement using local cache first, then Nominatim."""
    name = str(settlement or "").strip()
    if not name:
        return None
    key = name.lower().replace("ё", "е")
    if key in LOCAL_COORDINATES:
        lat, lon = LOCAL_COORDINATES[key]
        return lat, lon, "локальный справочник"

    query = ", ".join([part for part in [name, district, region, "Россия"] if str(part).strip()])
    url = (
        "https://nominatim.openstreetmap.org/search?format=json&limit=1&countrycodes=ru&q="
        + quote_plus(query)
    )
    request = Request(url, headers={"User-Agent": "udmurt-dialect-map/1.0"})
    try:
        with urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not data:
        return None
    first = data[0]
    return float(first["lat"]), float(first["lon"]), "Nominatim / OpenStreetMap"
