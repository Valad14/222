from __future__ import annotations

import colorsys
import hashlib
import json
import math
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from data_utils import add_unit_display, split_units


DEFAULT_REGION_BOUNDARIES: dict = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Удмуртская Республика"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [50.55, 55.80],
                        [54.55, 55.80],
                        [54.55, 58.55],
                        [50.55, 58.55],
                        [50.55, 55.80],
                    ]
                ],
            },
        }
    ],
}

SINGLE_POINT_COLOR_LABEL = "Все пункты"
SINGLE_POINT_FILL_COLOR = [37, 99, 235, 210]
SINGLE_POINT_OUTLINE_COLOR = [30, 64, 175, 255]

# Контрастная палитра: соседние категории получают заметно разные цвета.
DISTINCT_POINT_COLORS: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (0, 128, 128),
    (70, 80, 220),
    (220, 90, 150),
    (170, 110, 40),
    (128, 128, 0),
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (155, 89, 182),
    (243, 156, 18),
    (26, 188, 156),
    (44, 62, 80),
    (127, 140, 141),
]

LOCAL_COORDS: dict[tuple[str, str], tuple[float, float]] = {
    ("алнашский район", "алнаши"): (56.1873, 52.4794),
    ("балезинский район", "балезино"): (57.9796, 53.0138),
    ("вавожский район", "вавож"): (56.7752, 51.9300),
    ("граховский район", "грахово"): (56.0505, 51.9675),
    ("завьяловский район", "завьялово"): (56.7869, 53.3719),
    ("игринский район", "игра"): (57.5549, 53.0544),
    ("каракулинский район", "каракулино"): (56.0124, 53.7069),
    ("кезский район", "кез"): (57.8970, 53.7132),
    ("кизнерский район", "кизнер"): (56.2744, 51.5082),
    ("кияссовский район", "кияссово"): (56.3463, 53.1241),
    ("кияссовский район", "кияссово"): (56.3463, 53.1241),
    ("малопургинский район", "малая пурга"): (56.5569, 52.9958),
    ("можгинский район", "можга"): (56.4428, 52.2138),
    ("сарапульский район", "сарапул"): (56.4763, 53.7978),
    ("селтинский район", "селты"): (57.3121, 52.1310),
    ("сюмсинский район", "сюси"): (57.1115, 51.6158),
    ("сюмсинский район", "сюмси"): (57.1115, 51.6158),
    ("увинский район", "ува"): (56.9915, 52.1842),
    ("шарканский район", "шаркан"): (57.2986, 53.8712),
    ("якшур-бодьинский район", "якшур-бодья"): (57.1835, 53.1602),
}


def _clean_label(value: object, fallback: str = "нет данных") -> str:
    text = str(value or "").strip()
    return text or fallback


def label_color(label: object, alpha: int = 210) -> list[int]:
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
    return [int(red * 255), int(green * 255), int(blue * 255), alpha]


def label_color_hex(label: object) -> str:
    red, green, blue, _ = label_color(label, alpha=255)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _fallback_distinct_rgb(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
    return int(red * 255), int(green * 255), int(blue * 255)


def category_color_map(labels: Iterable[object], alpha: int = 210) -> dict[str, list[int]]:
    unique_labels = sorted({_clean_label(label) for label in labels}, key=lambda item: item.lower())
    colors: dict[str, list[int]] = {}
    for index, label in enumerate(unique_labels):
        rgb = DISTINCT_POINT_COLORS[index] if index < len(DISTINCT_POINT_COLORS) else _fallback_distinct_rgb(index)
        colors[label] = [rgb[0], rgb[1], rgb[2], alpha]
    return colors


def without_city_points(df: pd.DataFrame) -> pd.DataFrame:
    source = add_unit_display(df)
    if "settlement_type" not in source.columns:
        return source.copy()
    mask = ~source["settlement_type"].astype(str).str.lower().str.contains("город", na=False)
    return source.loc[mask].copy()


def _join_unique(values: Iterable[object], fallback: str = "") -> str:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() != "nan" and text not in result:
            result.append(text)
    return "; ".join(result) if result else fallback


def _compact_unique(values: Iterable[object], fallback: str = "не указан") -> str:
    result = [str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"]
    unique = []
    for item in result:
        if item not in unique:
            unique.append(item)
    if not unique:
        return fallback
    if len(unique) == 1:
        return unique[0]
    return "несколько"


def aggregate_points(map_df: pd.DataFrame) -> pd.DataFrame:
    source = add_unit_display(map_df).dropna(subset=["latitude", "longitude"]).copy()
    if source.empty:
        return pd.DataFrame()

    group_columns = ["region", "district", "settlement", "settlement_type", "latitude", "longitude"]
    rows: list[dict[str, object]] = []
    for keys, group in source.groupby(group_columns, dropna=False, sort=False):
        region, district, settlement, settlement_type, latitude, longitude = keys
        unit_display = _join_unique(group["unit_display"], fallback="нет данных")
        question = _join_unique(group["question"].head(4), fallback="")
        if group["question"].nunique() > 4:
            question += "; ..."
        row = {
            "region": region,
            "district": district,
            "settlement": settlement,
            "settlement_type": settlement_type,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "landscape": _compact_unique(group["landscape"], fallback="не указан"),
            "atlas_system": _compact_unique(group["atlas_system"], fallback="не указан"),
            "question_type": _compact_unique(group["question_type"], fallback="не указан"),
            "question": question,
            "unit_display": unit_display,
            "record_count": int(len(group)),
        }
        row["tooltip"] = (
            f"<b>{row['settlement']}</b><br>"
            f"{row['district']}, {row['region']}<br>"
            f"Записей: {row['record_count']}<br>"
            f"Единицы: {row['unit_display']}<br>"
            f"Ландшафт: {row['landscape']}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_point_visuals(points_df: pd.DataFrame, color_mode: str, single_color: bool = False) -> pd.DataFrame:
    result = points_df.copy()
    if result.empty:
        return result

    if single_color or color_mode == "Один цвет":
        result["color_label"] = SINGLE_POINT_COLOR_LABEL
        result["color"] = [SINGLE_POINT_FILL_COLOR for _ in range(len(result))]
        result["outline_color"] = [SINGLE_POINT_OUTLINE_COLOR for _ in range(len(result))]
    elif color_mode == "Ландшафт":
        result["color_label"] = result["landscape"].replace("", "не указан")
    elif color_mode == "Тип вопроса":
        result["color_label"] = result["question_type"].replace("", "не указан")
    elif color_mode == "Атлас":
        result["color_label"] = result["atlas_system"].replace("", "не указан")
    else:
        result["color_label"] = result["unit_display"].replace("", "нет данных")

    if not (single_color or color_mode == "Один цвет"):
        result["color_label"] = result["color_label"].map(_clean_label)
        fill_colors = category_color_map(result["color_label"], alpha=210)
        outline_colors = category_color_map(result["color_label"], alpha=255)
        result["color"] = result["color_label"].map(fill_colors)
        result["outline_color"] = result["color_label"].map(outline_colors)

    result["radius_m"] = 5200 + result["record_count"].clip(0, 10).astype(int) * 450
    result["short_label"] = result["settlement"].astype(str).str.slice(0, 18)
    return result


def map_view_state(points_df: pd.DataFrame) -> dict[str, float]:
    if points_df.empty:
        return {"latitude": 56.85, "longitude": 52.8, "zoom": 6.1, "pitch": 0}
    lat_span = float(points_df["latitude"].max() - points_df["latitude"].min())
    lon_span = float(points_df["longitude"].max() - points_df["longitude"].min())
    span = max(lat_span, lon_span)
    if span < 0.08:
        zoom = 10.0
    elif span < 0.4:
        zoom = 8.0
    elif span < 1.2:
        zoom = 6.9
    else:
        zoom = 6.0
    return {
        "latitude": float(points_df["latitude"].mean()),
        "longitude": float(points_df["longitude"].mean()),
        "zoom": zoom,
        "pitch": 0,
    }


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _buffered_polygon(points: list[tuple[float, float]], padding: float = 0.055) -> list[list[float]]:
    if not points:
        return []
    if len(points) == 1:
        lon, lat = points[0]
        return [
            [lon - padding, lat - padding],
            [lon + padding, lat - padding],
            [lon + padding, lat + padding],
            [lon - padding, lat + padding],
            [lon - padding, lat - padding],
        ]
    if len(points) == 2:
        (lon1, lat1), (lon2, lat2) = points
        dx = lon2 - lon1
        dy = lat2 - lat1
        length = math.hypot(dx, dy) or 1
        ox = -dy / length * padding
        oy = dx / length * padding
        return [
            [lon1 + ox, lat1 + oy],
            [lon2 + ox, lat2 + oy],
            [lon2 - ox, lat2 - oy],
            [lon1 - ox, lat1 - oy],
            [lon1 + ox, lat1 + oy],
        ]
    hull = _convex_hull(points)
    if hull and hull[0] != hull[-1]:
        hull.append(hull[0])
    return [[lon, lat] for lon, lat in hull]


def build_areals(exploded_df: pd.DataFrame, unit_column: str = "linguistic_unit") -> list[dict]:
    source = exploded_df.dropna(subset=["latitude", "longitude"]).copy()
    if source.empty or unit_column not in source.columns:
        return []

    areals: list[dict] = []
    for label, group in source.groupby(unit_column, dropna=False, sort=True):
        label_text = _clean_label(label)
        if label_text == "нет данных":
            continue
        points = [(float(row.longitude), float(row.latitude)) for row in group.itertuples()]
        unique_points = sorted(set(points))
        if not unique_points:
            continue
        polygon = _buffered_polygon(unique_points)
        if len(polygon) < 4:
            continue
        fill_color = label_color(label_text, alpha=42)
        line_color = label_color(label_text, alpha=185)
        areals.append(
            {
                "label": label_text,
                "polygon": polygon,
                "path": polygon,
                "fill_color": fill_color,
                "line_color": line_color,
            }
        )
    return areals


def geocode_settlement(settlement: str, district: str = "", region: str = "") -> tuple[float, float, str] | None:
    settlement_text = str(settlement or "").strip()
    district_text = str(district or "").strip()
    region_text = str(region or "").strip()
    if not settlement_text:
        return None

    local_key = (district_text.lower(), settlement_text.lower())
    if local_key in LOCAL_COORDS:
        lat, lon = LOCAL_COORDS[local_key]
        return lat, lon, "локальный справочник"

    query = ", ".join(part for part in [settlement_text, district_text, region_text, "Россия"] if part)
    url = f"https://nominatim.openstreetmap.org/search?format=json&limit=1&q={quote(query)}"
    request = Request(url, headers={"User-Agent": "dialect-map-streamlit/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload:
            return None
        first = payload[0]
        return float(first["lat"]), float(first["lon"]), "Nominatim / OpenStreetMap"
    except Exception:
        return None
