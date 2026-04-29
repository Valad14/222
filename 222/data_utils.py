from __future__ import annotations

import re
from io import StringIO
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ["region", "district", "settlement", "question"]
COORD_COLUMNS = ["latitude", "longitude"]

# В демо-наборе оставлено по 3 вопроса для ДАРЯ и ЛАРНГ.
DARYA_TYPES = [
    "ДАРЯ: фонетика",
    "ДАРЯ: морфология",
    "ДАРЯ: синтаксис",
]
LARNG_TYPES = [
    "ЛАРНГ: лексика / природа",
    "ЛАРНГ: лексика / материальная культура",
    "ЛАРНГ: лексика / человек",
]
ALLOWED_QUESTION_TYPES = DARYA_TYPES + LARNG_TYPES

CANONICAL_COLUMNS = [
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

DISPLAY_COLUMNS = [
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

TABLE_LABELS = {
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
    "unit_display": "Ответ / единица",
    "linguistic_unit_1": "Единица 1",
    "linguistic_unit_2": "Единица 2",
    "linguistic_unit_3": "Единица 3",
    "comment": "Комментарий",
    "source": "Источник",
    "year": "Год",
    "records": "Записей",
    "settlements": "Пунктов",
    "regions": "Регионов",
    "districts": "Районов",
    "units": "Единицы",
}

ALIASES = {
    "регион": "region",
    "область": "region",
    "край": "region",
    "республика": "region",
    "область / край / республика": "region",
    "область/край/республика": "region",
    "region": "region",
    "район": "district",
    "муниципальный район": "district",
    "district": "district",
    "населенный пункт": "settlement",
    "населённый пункт": "settlement",
    "н.п.": "settlement",
    "нп": "settlement",
    "settlement": "settlement",
    "тип населенного пункта": "settlement_type",
    "тип населённого пункта": "settlement_type",
    "settlement_type": "settlement_type",
    "широта": "latitude",
    "lat": "latitude",
    "latitude": "latitude",
    "долгота": "longitude",
    "lon": "longitude",
    "lng": "longitude",
    "longitude": "longitude",
    "ландшафт": "landscape",
    "landscape": "landscape",
    "атлас": "atlas_system",
    "источник вопросника": "atlas_system",
    "atlas": "atlas_system",
    "atlas_system": "atlas_system",
    "тип вопроса": "question_type",
    "раздел вопроса": "question_type",
    "question_type": "question_type",
    "код вопроса": "question_id",
    "номер вопроса": "question_id",
    "№ вопроса": "question_id",
    "question_id": "question_id",
    "вопрос": "question",
    "question": "question",
    "комментарий": "comment",
    "примечание": "comment",
    "comment": "comment",
    "источник": "source",
    "source": "source",
    "год": "year",
    "year": "year",
}


def _clean_name(name: object) -> str:
    text = str(name).strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def _alias_key(name: str) -> str:
    return _clean_name(name).lower().replace("ё", "е")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renamed: dict[str, str] = {}
    for col in df.columns:
        clean = _clean_name(col)
        key = _alias_key(clean)
        unit_match = re.match(
            r"^(лингвистическая\s+единица|единица|ответ|variant|unit|linguistic_unit)[ _-]*(\d+)$",
            key,
        )
        if unit_match:
            renamed[col] = f"linguistic_unit_{unit_match.group(2)}"
        elif key in ALIASES:
            renamed[col] = ALIASES[key]
        else:
            renamed[col] = clean
    return df.rename(columns=renamed)


def atlas_key(row_or_value: object, question_type: object | None = None) -> str:
    if isinstance(row_or_value, pd.Series):
        atlas = str(row_or_value.get("atlas_system", ""))
        qtype = str(row_or_value.get("question_type", ""))
    else:
        atlas = str(row_or_value or "")
        qtype = str(question_type or "")
    joined = f"{atlas} {qtype}".upper()
    if "ЛАР" in joined:
        return "ЛАРНГ"
    if "ДАР" in joined:
        return "ДАРЯ"
    return atlas.strip() or "не указан"


def infer_atlas_system(question_type: str, atlas_system: str = "") -> str:
    current = str(atlas_system or "").strip()
    if current and current != "не указан":
        return current
    key = atlas_key(current, question_type)
    return key if key in {"ДАРЯ", "ЛАРНГ"} else "не указан"


def get_linguistic_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if re.match(r"^linguistic_unit_\d+$", str(col)):
            cols.append(str(col))
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def normalize_question_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy question codes with simple 1, 2, 3 numbering.

    Numbering is done separately for ДАРЯ and ЛАРНГ, so each system can have
    exactly three compact question numbers in the demo data.
    """
    df = df.copy()
    if df.empty:
        return df

    mapping: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    for _, row in df.iterrows():
        system = atlas_key(row)
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        key = (system, question)
        if key not in mapping:
            counters[system] = counters.get(system, 0) + 1
            mapping[key] = str(counters[system])

    def _number(row: pd.Series) -> str:
        question = str(row.get("question", "")).strip()
        if not question:
            return str(row.get("question_id", "")).strip()
        return mapping.get((atlas_key(row), question), str(row.get("question_id", "")).strip())

    df["question_id"] = df.apply(_number, axis=1)
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    text_cols = [
        "region",
        "district",
        "settlement",
        "settlement_type",
        "landscape",
        "atlas_system",
        "question_type",
        "question_id",
        "question",
        "comment",
        "source",
        "year",
    ]
    for text_col in text_cols:
        df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    for col in get_linguistic_columns(df):
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["atlas_system"] = df.apply(
        lambda row: infer_atlas_system(row["question_type"], row["atlas_system"]), axis=1
    )
    df["question_type"] = df["question_type"].replace({"": "не указан"})
    df["landscape"] = df["landscape"].replace({"": "не указан"})
    df["settlement_type"] = df["settlement_type"].replace({"": "населённый пункт"})
    df = normalize_question_ids(df)
    return df


def read_csv_bytes(content: bytes, sep: str | None = None) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            text = content.decode(encoding)
            return ensure_columns(pd.read_csv(StringIO(text), sep=sep or None, engine="python"))
        except Exception:
            continue
    raise ValueError("Не удалось прочитать CSV. Проверьте кодировку UTF-8/CP1251 и разделитель.")


def read_csv_path(path: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return ensure_columns(pd.read_csv(path, encoding=encoding))
        except Exception:
            continue
    raise ValueError(f"Не удалось прочитать файл: {path}")


def read_csv_url(url: str) -> pd.DataFrame:
    return ensure_columns(pd.read_csv(url))


def split_units(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    parts = re.split(r";|,|\|", str(value))
    return [p.strip() for p in parts if p and p.strip()]


def row_units(row: pd.Series, unit_cols: Iterable[str] | None = None) -> list[str]:
    if unit_cols is None:
        unit_cols = [c for c in row.index if re.match(r"^linguistic_unit_\d+$", str(c))]
    units: list[str] = []
    for col in unit_cols:
        units.extend(split_units(row.get(col, "")))
    seen: set[str] = set()
    result: list[str] = []
    for unit in units:
        key = unit.lower()
        if key not in seen:
            seen.add(key)
            result.append(unit)
    return result


def add_unit_display(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df).copy()
    unit_cols = get_linguistic_columns(df)
    df["unit_display"] = df.apply(lambda row: "; ".join(row_units(row, unit_cols)), axis=1)
    df["unit_display"] = df["unit_display"].replace({"": "нет данных"})
    return df


def explode_units(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df)
    unit_cols = get_linguistic_columns(df)
    records: list[dict] = []
    for _, row in df.iterrows():
        units = row_units(row, unit_cols) or ["нет данных"]
        for unit in units:
            item = row.to_dict()
            item["linguistic_unit"] = unit
            records.append(item)
    return pd.DataFrame(records)


def get_all_units(df: pd.DataFrame) -> list[str]:
    units: set[str] = set()
    for _, row in ensure_columns(df).iterrows():
        for unit in row_units(row):
            if unit:
                units.add(unit)
    return sorted(units, key=lambda item: item.lower())


def question_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = add_unit_display(ensure_columns(df))
    grouped = (
        df.groupby(["atlas_system", "question_id", "question", "question_type"], dropna=False, sort=False)
        .agg(
            records=("settlement", "size"),
            settlements=("settlement", "nunique"),
            regions=("region", "nunique"),
            districts=("district", "nunique"),
            units=("unit_display", lambda s: "; ".join(sorted({u for v in s for u in split_units(v)}))),
        )
        .reset_index()
    )
    grouped["sort_system"] = grouped["atlas_system"].map({"ДАРЯ": 0, "ЛАРНГ": 1}).fillna(9)
    grouped["sort_id"] = pd.to_numeric(grouped["question_id"], errors="coerce").fillna(9999)
    grouped = grouped.sort_values(["sort_system", "sort_id", "question"]).drop(columns=["sort_system", "sort_id"])
    return grouped


def question_key(atlas_system: object, question: object) -> str:
    return f"{atlas_system}||{question}"


def filter_dataframe(
    df: pd.DataFrame,
    regions: list[str] | None = None,
    districts: list[str] | None = None,
    question: str | None = None,
    unit_query: str | None = None,
    text_query: str | None = None,
) -> pd.DataFrame:
    df = add_unit_display(ensure_columns(df))
    if regions:
        df = df[df["region"].isin(regions)]
    if districts:
        df = df[df["district"].isin(districts)]
    if question and question != "Все вопросы":
        if "||" in question:
            atlas, question_text = question.split("||", 1)
            df = df[(df["atlas_system"] == atlas) & (df["question"] == question_text)]
        else:
            df = df[(df["question"] == question) | (df["question_id"] == question)]
    if unit_query:
        query = unit_query.strip().lower()
        if query:
            df = df[df["unit_display"].str.lower().str.contains(re.escape(query), na=False)]
    if text_query:
        query = text_query.strip().lower()
        if query:
            haystack = (
                df["region"]
                + " "
                + df["district"]
                + " "
                + df["settlement"]
                + " "
                + df["settlement_type"]
                + " "
                + df["landscape"]
                + " "
                + df["question_type"]
                + " "
                + df["question"]
                + " "
                + df["unit_display"]
                + " "
                + df["comment"]
            ).str.lower()
            df = df[haystack.str.contains(re.escape(query), na=False)]
    return df


def validate_dataframe(df: pd.DataFrame) -> list[dict[str, str]]:
    df = ensure_columns(df)
    issues: list[dict[str, str]] = []
    for col in REQUIRED_COLUMNS:
        empty_count = int((df[col].fillna("").astype(str).str.strip() == "").sum())
        if empty_count:
            issues.append({"level": "Ошибка", "field": col, "message": f"Пустых значений: {empty_count}"})

    no_coords = int(df[["latitude", "longitude"]].isna().any(axis=1).sum())
    if no_coords:
        issues.append(
            {
                "level": "Предупреждение",
                "field": "latitude/longitude",
                "message": f"Записей без координат: {no_coords}; они не появятся на карте.",
            }
        )

    city_rows = int(df["settlement_type"].str.lower().str.contains("город|^г\\.?$", regex=True, na=False).sum())
    if city_rows:
        issues.append(
            {
                "level": "Предупреждение",
                "field": "settlement_type",
                "message": f"Строк с типом город: {city_rows}. Такие пункты скрываются на карте.",
            }
        )

    invalid_question_types = sorted(
        {
            q
            for q in df["question_type"].dropna().unique()
            if q and q != "не указан" and q not in ALLOWED_QUESTION_TYPES
        }
    )
    if invalid_question_types:
        issues.append(
            {
                "level": "Предупреждение",
                "field": "question_type",
                "message": "Нетиповые разделы: " + "; ".join(invalid_question_types),
            }
        )

    unit_cols = get_linguistic_columns(df)
    if not unit_cols:
        issues.append(
            {
                "level": "Ошибка",
                "field": "linguistic_unit_*",
                "message": "Добавьте хотя бы один столбец linguistic_unit_1 / «лингвистическая единица 1».",
            }
        )
    else:
        empty_units = int(df[unit_cols].replace("", pd.NA).isna().all(axis=1).sum())
        if empty_units:
            issues.append(
                {
                    "level": "Предупреждение",
                    "field": "linguistic_unit_*",
                    "message": f"Записей без лингвистических единиц: {empty_units}.",
                }
            )

    duplicate_rows = int(df.duplicated(subset=["region", "district", "settlement", "atlas_system", "question"]).sum())
    if duplicate_rows:
        issues.append(
            {
                "level": "Предупреждение",
                "field": "duplicates",
                "message": f"Возможных дублей пункт-вопрос: {duplicate_rows}.",
            }
        )
    return issues


def display_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    df = add_unit_display(df)
    columns = columns or DISPLAY_COLUMNS
    available = [col for col in columns if col in df.columns]
    result = df[available].copy()
    return result.rename(columns={col: TABLE_LABELS.get(col, col) for col in available})


def to_download_csv(df: pd.DataFrame) -> bytes:
    return ensure_columns(df).to_csv(index=False).encode("utf-8-sig")
