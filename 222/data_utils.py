from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS: list[str] = [
    "region",
    "district",
    "settlement",
    "settlement_type",
    "latitude",
    "longitude",
    "landscape",
    "atlas_system",
    "question_type",
    "question_id",
    "question",
    "linguistic_unit_1",
    "linguistic_unit_2",
    "linguistic_unit_3",
    "comment",
    "source",
    "year",
]

DISPLAY_COLUMNS: list[str] = [
    "region",
    "district",
    "settlement",
    "settlement_type",
    "latitude",
    "longitude",
    "landscape",
    "question_type",
    "question_id",
    "question",
    "unit_display",
    "comment",
]

TABLE_LABELS: dict[str, str] = {
    "region": "Регион",
    "district": "Район",
    "settlement": "Населённый пункт",
    "settlement_type": "Тип пункта",
    "latitude": "Широта",
    "longitude": "Долгота",
    "landscape": "Ландшафт",
    "atlas_system": "Атлас",
    "question_type": "Раздел",
    "question_id": "№ вопроса",
    "question": "Вопрос",
    "unit_display": "Диалектные единицы",
    "linguistic_unit_1": "Единица 1",
    "linguistic_unit_2": "Единица 2",
    "linguistic_unit_3": "Единица 3",
    "comment": "Комментарий",
    "source": "Источник",
    "year": "Год",
    "settlements": "Пунктов",
    "regions": "Регионов",
    "districts": "Районов",
    "units": "Единицы",
}

ALLOWED_QUESTION_TYPES: list[str] = [
    "ДАРЯ: фонетика",
    "ДАРЯ: морфология",
    "ДАРЯ: синтаксис",
    "ЛАРНГ: лексика / природа",
    "ЛАРНГ: лексика / человек",
    "ЛАРНГ: лексика / материальная культура",
]


def unit_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("linguistic_unit")]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _clean_question_id(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # Remove accidental BOM from the first column name and surrounding spaces.
    result.columns = [str(column).replace("\ufeff", "").strip() for column in result.columns]

    for column in CANONICAL_COLUMNS:
        if column not in result.columns:
            result[column] = ""

    for column in unit_columns(result):
        result[column] = result[column].map(_clean_text)

    text_columns = [column for column in result.columns if column not in {"latitude", "longitude"}]
    for column in text_columns:
        result[column] = result[column].map(_clean_text)

    result["question_id"] = result["question_id"].map(_clean_question_id)
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")

    return add_unit_display(result)


def read_csv_path(path: str) -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(path, encoding="utf-8-sig"))


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(BytesIO(data), encoding="utf-8-sig"))


def read_csv_url(url: str) -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(url, encoding="utf-8-sig"))


def split_units(value: object) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    parts: list[str] = []
    for piece in text.replace("|", ";").split(";"):
        cleaned = piece.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def row_units(row: pd.Series, columns: Iterable[str] | None = None) -> list[str]:
    cols = list(columns) if columns is not None else [c for c in row.index if str(c).startswith("linguistic_unit")]
    units: list[str] = []
    for column in cols:
        for unit in split_units(row.get(column, "")):
            if unit not in units:
                units.append(unit)
    return units


def add_unit_display(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    columns = unit_columns(result)
    if not columns:
        result["unit_display"] = ""
    else:
        result["unit_display"] = result.apply(lambda row: "; ".join(row_units(row, columns)), axis=1)

    search_columns = [
        column
        for column in [
            "region",
            "district",
            "settlement",
            "settlement_type",
            "landscape",
            "atlas_system",
            "question_type",
            "question_id",
            "question",
            "unit_display",
            "comment",
            "source",
            "year",
        ]
        if column in result.columns
    ]
    result["search_blob"] = result[search_columns].astype(str).agg(" ".join, axis=1).str.lower()
    return result


def question_key(atlas_system: object, question: object) -> str:
    return f"{_clean_text(atlas_system)}||{_clean_text(question)}"


def question_catalog(df: pd.DataFrame) -> pd.DataFrame:
    source = add_unit_display(df)
    group_columns = ["atlas_system", "question_type", "question_id", "question"]
    if source.empty:
        return pd.DataFrame(columns=group_columns + ["settlements", "regions", "districts", "units", "key"])

    rows: list[dict[str, object]] = []
    for group_values, group in source.groupby(group_columns, dropna=False, sort=False):
        atlas, qtype, qid, question = group_values
        units = sorted({unit for value in group["unit_display"] for unit in split_units(value)}, key=lambda x: x.lower())
        rows.append(
            {
                "atlas_system": atlas,
                "question_type": qtype,
                "question_id": qid,
                "question": question,
                "settlements": int(group["settlement"].nunique()),
                "regions": int(group["region"].nunique()),
                "districts": int(group["district"].nunique()),
                "units": "; ".join(units),
                "key": question_key(atlas, question),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["atlas_system", "question_type", "question_id", "question"], kind="stable").reset_index(drop=True)


def filter_dataframe(
    df: pd.DataFrame,
    regions: list[str] | None = None,
    districts: list[str] | None = None,
    question: str = "Все вопросы",
    unit_query: str = "",
    text_query: str = "",
) -> pd.DataFrame:
    result = add_unit_display(df)

    if regions:
        result = result[result["region"].isin(regions)]
    if districts:
        result = result[result["district"].isin(districts)]
    if question and question != "Все вопросы":
        keys = result.apply(lambda row: question_key(row.get("atlas_system", ""), row.get("question", "")), axis=1)
        result = result[keys == question]

    unit_query = str(unit_query or "").strip().lower()
    if unit_query:
        result = result[result["unit_display"].str.lower().str.contains(unit_query, regex=False, na=False)]

    text_query = str(text_query or "").strip().lower()
    if text_query:
        result = result[result["search_blob"].str.contains(text_query, regex=False, na=False)]

    return result.copy()


def explode_units(df: pd.DataFrame) -> pd.DataFrame:
    source = add_unit_display(df)
    rows: list[dict[str, object]] = []
    columns = unit_columns(source)
    for _, row in source.iterrows():
        units = row_units(row, columns)
        if not units:
            units = [""]
        for unit in units:
            item = row.to_dict()
            item["linguistic_unit"] = unit
            rows.append(item)
    return pd.DataFrame(rows)


def get_all_units(df: pd.DataFrame) -> list[str]:
    source = add_unit_display(df)
    units = {unit for value in source["unit_display"] for unit in split_units(value)}
    return sorted(units, key=lambda x: x.lower())


def display_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    source = add_unit_display(df)
    selected = columns or DISPLAY_COLUMNS
    existing = [column for column in selected if column in source.columns]
    result = source.loc[:, existing].copy()
    return result.rename(columns={column: TABLE_LABELS.get(column, column) for column in result.columns})


def to_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def validate_dataframe(df: pd.DataFrame) -> list[dict[str, object]]:
    source = add_unit_display(df)
    issues: list[dict[str, object]] = []
    required = ["region", "district", "settlement", "latitude", "longitude", "question_type", "question", "unit_display"]

    for index, row in source.iterrows():
        row_number = int(index) + 2
        for column in required:
            value = row.get(column, "")
            if column in {"latitude", "longitude"}:
                if pd.isna(value):
                    issues.append({"Строка": row_number, "Поле": TABLE_LABELS.get(column, column), "Проблема": "нет координаты"})
            elif not _clean_text(value):
                issues.append({"Строка": row_number, "Поле": TABLE_LABELS.get(column, column), "Проблема": "пустое обязательное поле"})

        lat = row.get("latitude")
        lon = row.get("longitude")
        if not pd.isna(lat) and not (-90 <= float(lat) <= 90):
            issues.append({"Строка": row_number, "Поле": "Широта", "Проблема": "значение вне диапазона -90..90"})
        if not pd.isna(lon) and not (-180 <= float(lon) <= 180):
            issues.append({"Строка": row_number, "Поле": "Долгота", "Проблема": "значение вне диапазона -180..180"})

    return issues
