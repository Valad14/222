from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pydeck as pdk
import streamlit as st

from data_utils import (
    ALLOWED_QUESTION_TYPES,
    CANONICAL_COLUMNS,
    DISPLAY_COLUMNS,
    TABLE_LABELS,
    add_unit_display,
    display_dataframe,
    explode_units,
    filter_dataframe,
    get_all_units,
    question_catalog,
    question_key,
    read_csv_bytes,
    read_csv_path,
    read_csv_url,
    to_download_csv,
    validate_dataframe,
)
from geo_utils import (
    DEFAULT_REGION_BOUNDARIES,
    add_point_visuals,
    aggregate_points,
    build_areals,
    geocode_settlement,
    label_color_hex,
    map_view_state,
    without_city_points,
)

APP_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = APP_DIR / "data" / "sample_dialects.csv"
TEMPLATE_DATA_PATH = APP_DIR / "data" / "data_template.csv"
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return read_csv_path(str(SAMPLE_DATA_PATH))


@st.cache_data(show_spinner=False)
def cached_url_csv(url: str) -> pd.DataFrame:
    return read_csv_url(url)


@st.cache_data(show_spinner=False)
def cached_geojson_url(url: str) -> dict:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(show_spinner=False)
def read_template_bytes() -> bytes:
    return TEMPLATE_DATA_PATH.read_bytes()


def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def inject_css() -> None:
    """Force readable dark text on white backgrounds for filters, legend and editors."""
    st.markdown(
        """
<style>
:root {
    --text-main: #0f172a;
    --text-muted: #475569;
    --border-soft: #cbd5e1;
    --border-soft-2: #e2e8f0;
    --panel-bg: #ffffff;
    --page-bg: #f8fafc;
    --accent-soft: #ffffff;
}

html, body, .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--page-bg) !important;
    color: var(--text-main) !important;
}

[data-testid="stToolbar"] { background: transparent !important; }

section[data-testid="stSidebar"],
[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-right: 1px solid var(--border-soft) !important;
    color: var(--text-main) !important;
}

section[data-testid="stSidebar"] *,
[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] label {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
}

h1, h2, h3, h4, h5, h6, p, li, label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
.stAlert, .stAlert * {
    color: var(--text-main) !important;
}

/* White fields: filters, text inputs, selectboxes, multiselects and text areas. */
[data-baseweb="input"],
[data-baseweb="select"],
[data-baseweb="textarea"],
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    -webkit-text-fill-color: var(--text-main) !important;
}

[data-baseweb="input"] svg,
[data-baseweb="select"] svg,
[data-baseweb="textarea"] svg {
    color: var(--text-main) !important;
    fill: var(--text-main) !important;
}

/* Streamlit renders dropdown menus outside the sidebar, so style global popovers too. */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"],
li[role="option"],
div[role="option"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: var(--text-main) !important;
}

li[role="option"]:hover,
div[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: var(--text-main) !important;
}

/* Selected values in multiselect. */
[data-baseweb="tag"],
[data-baseweb="tag"] span,
[data-baseweb="tag"] svg {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    fill: var(--text-main) !important;
}

.stCheckbox,
.stCheckbox label,
[data-testid="stCheckbox"],
[data-testid="stPopover"],
[data-testid="stPopover"] button,
[data-testid="stPopoverBody"],
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: var(--text-main) !important;
}

.stButton > button,
.stDownloadButton > button,
.stLinkButton > a,
button[kind="secondary"],
button[kind="primary"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}

[data-testid="stDataFrame"],
[data-testid="stDataEditor"],
[data-testid="stDataFrame"] *,
[data-testid="stDataEditor"] *,
.stDataFrame,
.stDataEditor {
    background-color: #ffffff !important;
    color: var(--text-main) !important;
}

.clean-table-wrap {
    max-height: 430px;
    overflow: auto;
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 4px 18px rgba(15, 23, 42, .06);
}

table.clean-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 13px;
    line-height: 1.4;
}

table.clean-table th {
    position: sticky;
    top: 0;
    z-index: 1;
    padding: 9px 10px;
    border-bottom: 1px solid var(--border-soft);
    background: #ffffff !important;
    color: #0f172a !important;
    text-align: left;
    font-weight: 700;
    vertical-align: top;
}

table.clean-table td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border-soft-2);
    color: #0f172a !important;
    vertical-align: top;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
    background: #ffffff !important;
}

table.clean-table tr:nth-child(even) td { background: #ffffff !important; }

.badge {
    display: inline-block;
    border: 1px solid #bae6fd;
    background: #ffffff !important;
    color: #0f172a !important;
    border-radius: 999px;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
    font-size: 12px;
}

.small-card, .section-card {
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 14px 16px;
    background: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 4px 18px rgba(15, 23, 42, .06);
}

.section-card { margin-bottom: 1rem; }

hr {
    border: none;
    border-top: 1px solid var(--border-soft-2);
    margin: 1.25rem 0 1rem 0;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _table_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return html.escape(text)


def render_light_dataframe(df: pd.DataFrame, height: int = 430, max_rows: int = 500) -> None:
    if df.empty:
        st.info("В таблице нет записей для выбранных фильтров.")
        return

    shown = df.head(max_rows).copy()
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in shown.columns)
    rows: list[str] = []
    for _, row in shown.iterrows():
        rows.append("<tr>" + "".join(f"<td>{_table_cell(row[col])}</td>" for col in shown.columns) + "</tr>")

    note = ""
    if len(df) > max_rows:
        note = f"<p style='margin:8px 2px;color:#64748b'>Показано {max_rows} из {len(df)} строк.</p>"

    st.markdown(
        f"<div class='clean-table-wrap' style='max-height:{height}px'>"
        f"<table class='clean-table'><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{note}",
        unsafe_allow_html=True,
    )


def load_data_from_sidebar() -> tuple[pd.DataFrame, str, str]:
    st.sidebar.markdown("## Источник данных")
    if st.sidebar.button("Обновить кэш данных"):
        st.cache_data.clear()
        st.sidebar.success("Данные будут перечитаны.")

    secret_url = _secret("DATA_CSV_URL")
    secret_edit_url = _secret("REMOTE_TABLE_EDIT_URL")
    source_kind = st.sidebar.radio(
        "Данные",
        ["Демо-данные", "Google Sheets / CSV URL", "Загрузить CSV"],
        index=0,
    )
    editor_url = secret_edit_url

    if source_kind == "Google Sheets / CSV URL":
        url = st.sidebar.text_input(
            "CSV URL",
            value=secret_url,
            help="Подходит опубликованный CSV Google Sheets или любой HTTPS CSV.",
        )
        editor_url = st.sidebar.text_input(
            "Ссылка на редактирование таблицы",
            value=secret_edit_url,
            help="Необязательно: ссылка на Google Sheets для редакторов.",
        )
        if not url:
            st.sidebar.warning("Вставьте URL CSV или задайте DATA_CSV_URL в secrets.")
            return load_sample_data(), "Демо-данные: URL не задан", editor_url
        try:
            return cached_url_csv(url), "Удалённая таблица CSV/Google Sheets", editor_url
        except Exception as exc:
            st.sidebar.error(f"Не удалось загрузить URL: {exc}")
            return load_sample_data(), "Демо-данные: ошибка URL", editor_url

    if source_kind == "Загрузить CSV":
        uploaded = st.sidebar.file_uploader("CSV-файл", type=["csv"])
        if uploaded is not None:
            try:
                return read_csv_bytes(uploaded.getvalue()), f"Загружен файл: {uploaded.name}", editor_url
            except Exception as exc:
                st.sidebar.error(str(exc))
        return load_sample_data(), "Демо-данные: CSV не загружен", editor_url

    return load_sample_data(), "Демо-данные из пакета", editor_url


def get_working_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    added_records = st.session_state.get("added_records", [])
    if not added_records:
        return df
    added_df = pd.DataFrame(added_records)
    combined = pd.concat([df, added_df], ignore_index=True, sort=False)
    return add_unit_display(combined)


def sidebar_filters(df: pd.DataFrame) -> dict:
    st.sidebar.markdown("## Фильтры карты")
    map_df = without_city_points(add_unit_display(df))

    regions = sorted([x for x in map_df["region"].dropna().unique() if x])
    selected_regions = st.sidebar.multiselect("Регионы", regions, default=regions)

    district_pool = map_df[map_df["region"].isin(selected_regions)] if selected_regions else map_df
    districts = sorted([x for x in district_pool["district"].dropna().unique() if x])
    selected_districts = st.sidebar.multiselect("Районы / округа", districts, default=[])

    catalog = question_catalog(map_df)
    question_labels = ["Все вопросы"]
    label_to_key: dict[str, str] = {"Все вопросы": "Все вопросы"}
    for row in catalog.itertuples():
        label = f"{row.question_type} · {row.question_id}. {row.question}"
        question_labels.append(label)
        label_to_key[label] = question_key(row.atlas_system, row.question)

    selected_label = st.sidebar.selectbox("Карта / вопрос", question_labels, index=0)
    unit_query = st.sidebar.text_input("Поиск лингвистической единицы", placeholder="например: [ɣ], ляда, -ут")
    text_query = st.sidebar.text_input("Общий поиск", placeholder="пункт, район, комментарий")
    color_mode = st.sidebar.selectbox(
        "Раскраска точек",
        ["Диалектные единицы", "Ландшафт", "Тип вопроса", "Атлас"],
        index=0,
    )
    show_areals = st.sidebar.checkbox("Показывать ареалы", value=True)
    show_isoglosses = st.sidebar.checkbox("Показывать изоглоссы", value=True)
    show_labels = st.sidebar.checkbox("Подписи пунктов", value=False)

    return {
        "regions": selected_regions,
        "districts": selected_districts,
        "selected_question": label_to_key[selected_label],
        "unit_query": unit_query,
        "text_query": text_query,
        "color_mode": color_mode,
        "show_areals": show_areals,
        "show_isoglosses": show_isoglosses,
        "show_labels": show_labels,
    }


def get_boundaries() -> dict:
    url = _secret("BOUNDARY_GEOJSON_URL")
    if url:
        try:
            return cached_geojson_url(url)
        except Exception as exc:
            st.warning(f"GeoJSON-слой из URL не загружен, показаны встроенные границы регионов: {exc}")
    return DEFAULT_REGION_BOUNDARIES


def make_deck(
    df: pd.DataFrame,
    selected_question: str = "Все вопросы",
    color_mode: str = "Диалектные единицы",
    show_areals: bool = True,
    show_isoglosses: bool = True,
    show_labels: bool = False,
    force_areals: bool = False,
    height: int = 640,
) -> tuple[pdk.Deck | None, pd.DataFrame, list[dict]]:
    df = add_unit_display(df)
    map_df = without_city_points(df)
    points_df = aggregate_points(map_df)
    if points_df.empty:
        return None, points_df, []

    points_df = add_point_visuals(points_df, color_mode)
    layers: list[pdk.Layer] = []
    areals: list[dict] = []
    should_build_areals = selected_question != "Все вопросы" or force_areals

    if should_build_areals:
        exploded = explode_units(map_df)
        areals = build_areals(exploded, "linguistic_unit")
        if show_areals and areals:
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=areals,
                    get_polygon="polygon",
                    get_fill_color="fill_color",
                    get_line_color="line_color",
                    line_width_min_pixels=1,
                    stroked=True,
                    filled=True,
                    pickable=False,
                    auto_highlight=False,
                )
            )
        if show_isoglosses and areals:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=areals,
                    get_path="path",
                    get_color="line_color",
                    width_scale=1,
                    width_min_pixels=3,
                    pickable=False,
                )
            )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=points_df,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_line_color="outline_color",
            get_radius="radius_m",
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            radius_min_pixels=5,
            radius_max_pixels=24,
            pickable=True,
            auto_highlight=True,
        )
    )

    if show_labels:
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=points_df,
                get_position="[longitude, latitude]",
                get_text="short_label",
                get_size=13,
                get_color=[15, 23, 42, 230],
                get_alignment_baseline="'bottom'",
                get_pixel_offset=[0, -12],
                pickable=False,
            )
        )

    deck = pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=pdk.ViewState(**map_view_state(points_df)),
        layers=layers,
        tooltip={
            "html": "{tooltip}<br/>Цвет: {color_label}",
            "style": {
                "backgroundColor": "#ffffff",
                "color": "#0f172a",
                "fontSize": "12px",
                "border": "1px solid #cbd5e1",
                "boxShadow": "0 4px 18px rgba(15, 23, 42, .14)",
            },
        },
        height=height,
    )
    return deck, points_df, areals


def render_legend(points_df: pd.DataFrame, areals: list[dict]) -> None:
    if points_df.empty:
        st.info("Легенда появится, когда на карте будут точки.")
        return

    labels = (
        points_df.groupby("color_label")
        .agg(points=("settlement", "nunique"))
        .reset_index()
        .sort_values(["points", "color_label"], ascending=[False, True])
    )
    summary = f"{len(labels)} категорий"
    if areals:
        summary += f" · {len(areals)} ареалов"

    legend_html = [f"<div class='section-card'><b>{html.escape(summary)}</b><br/>"]
    for _, row in labels.head(12).iterrows():
        label = str(row["color_label"])
        color = label_color_hex(label)
        legend_html.append(
            f"<span class='badge'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{color};margin-right:6px;border:1px solid #94a3b8'></span>"
            f"{html.escape(label)} — {int(row['points'])}</span>"
        )
    if len(labels) > 12:
        legend_html.append(f"<p style='margin:8px 0 0 0'>Показано 12 из {len(labels)} значений.</p>")
    if areals:
        legend_html.append("<p style='margin:8px 0 0 0'>Ареалы построены по текущей выборке, включая общий поиск.</p>")
    legend_html.append("</div>")

    container = st.popover("Легенда") if hasattr(st, "popover") else st.expander("Легенда", expanded=True)
    with container:
        st.markdown("\n".join(legend_html), unsafe_allow_html=True)


def render_atlas_page(df: pd.DataFrame, filters: dict) -> None:
    st.subheader("Карта: пункты, ареалы, изоглоссы")
    filtered = filter_dataframe(
        df,
        regions=filters["regions"],
        districts=filters["districts"],
        question=filters["selected_question"],
        unit_query=filters["unit_query"],
        text_query=filters["text_query"],
    )
    force_areals = bool(filters["text_query"].strip() or filters["unit_query"].strip())
    left, right = st.columns([3.2, 1.15], gap="large")
    with left:
        deck, points_df, areals = make_deck(
            filtered,
            selected_question=filters["selected_question"],
            color_mode=filters["color_mode"],
            show_areals=filters["show_areals"],
            show_isoglosses=filters["show_isoglosses"],
            show_labels=filters["show_labels"],
            force_areals=force_areals,
        )
        if deck is None:
            st.warning("Нет не городских пунктов с координатами для выбранных фильтров.")
        else:
            st.pydeck_chart(deck, use_container_width=True)
    with right:
        st.caption("Компактная легенда")
        render_legend(points_df if "points_df" in locals() else pd.DataFrame(), areals if "areals" in locals() else [])
        st.metric("Пунктов на карте", int(points_df["settlement"].nunique()) if "points_df" in locals() and not points_df.empty else 0)
        st.metric("Ареалов", len(areals) if "areals" in locals() else 0)
    with st.expander("Таблица выбранных записей", expanded=False):
        render_light_dataframe(display_dataframe(filtered, DISPLAY_COLUMNS), height=460)
    st.download_button(
        "Скачать выбранные записи CSV",
        data=to_download_csv(filtered),
        file_name="dialekt_selected_records.csv",
        mime="text/csv",
    )


def render_maps_page(df: pd.DataFrame) -> None:
    st.subheader("Поиск и демонстрация карт")
    catalog = question_catalog(without_city_points(df))
    q = st.text_input("Поиск карты по номеру, вопросу, разделу или системе", placeholder="ДАРЯ фонетика / дождь / 1")
    if q:
        needle = q.lower()
        visible = catalog[catalog.apply(lambda row: needle in " ".join(map(str, row.values)).lower(), axis=1)].copy()
    else:
        visible = catalog.copy()

    catalog_view = visible[["question_id", "question_type", "question", "settlements", "regions", "districts", "units"]]
    render_light_dataframe(
        catalog_view.rename(columns={col: TABLE_LABELS.get(col, col) for col in catalog_view.columns}),
        height=300,
    )
    if visible.empty:
        st.info("Поиск не дал результатов.")
        return

    labels: list[str] = []
    label_to_question: dict[str, str] = {}
    for row in visible.itertuples():
        label = f"{row.question_type} · {row.question_id}. {row.question}"
        labels.append(label)
        label_to_question[label] = question_key(row.atlas_system, row.question)

    selected = st.selectbox("Открыть карту", labels)
    subset = filter_dataframe(df, question=label_to_question[selected])
    st.markdown("### Демонстрация выбранной карты")
    left, right = st.columns([3, 1.2], gap="large")
    with left:
        deck, points_df, areals = make_deck(
            subset,
            selected_question=label_to_question[selected],
            color_mode="Диалектные единицы",
            show_areals=True,
            show_isoglosses=True,
            show_labels=True,
            height=520,
        )
        if deck:
            st.pydeck_chart(deck, use_container_width=True)
    with right:
        passport = question_catalog(subset).iloc[0]
        st.markdown(
            f"""
<div class='section-card'>
<b>Паспорт карты</b><br/>
№ вопроса: {html.escape(str(passport['question_id']))}<br/>
Раздел: {html.escape(str(passport['question_type']))}<br/>
Пунктов: {int(passport['settlements'])}<br/>
Единицы: {html.escape(str(passport['units']))}
</div>
""",
            unsafe_allow_html=True,
        )
        render_legend(points_df if "points_df" in locals() else pd.DataFrame(), areals if "areals" in locals() else [])
    st.download_button("Скачать данные этой карты", data=to_download_csv(subset), file_name="map_data.csv", mime="text/csv")


def render_points_page(df: pd.DataFrame) -> None:
    st.subheader("Пункты, районы и регионы")
    map_df = without_city_points(add_unit_display(df))
    c1, c2, c3 = st.columns(3)
    with c1:
        region = st.selectbox("Регион", ["Все регионы"] + sorted(map_df["region"].dropna().unique()))
        region_df = map_df if region == "Все регионы" else map_df[map_df["region"] == region]
    with c2:
        district = st.selectbox("Район", ["Все районы"] + sorted(region_df["district"].dropna().unique()))
        district_df = region_df if district == "Все районы" else region_df[region_df["district"] == district]
    with c3:
        settlement_query = st.text_input("Поиск населённого пункта", placeholder="Алнаши, Ува...")
    if settlement_query:
        district_df = district_df[district_df["settlement"].str.lower().str.contains(settlement_query.lower(), na=False)]

    settlements = sorted(district_df["settlement"].dropna().unique())
    if not settlements:
        st.info("По выбранным условиям пунктов нет.")
        return

    selected_settlement = st.selectbox("Открыть пункт", settlements)
    point_df = district_df[district_df["settlement"] == selected_settlement]
    left, right = st.columns([1.15, 2.3], gap="large")
    first = point_df.iloc[0]
    with left:
        st.markdown(
            f"""
<div class='section-card'>
<b>{html.escape(str(first['settlement']))}</b><br/>
{html.escape(str(first['settlement_type']))}<br/>
{html.escape(str(first['district']))}, {html.escape(str(first['region']))}<br/>
Ландшафт: {html.escape(str(first['landscape']))}<br/>
Координаты: {first['latitude']:.5f}, {first['longitude']:.5f}
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        deck, _, _ = make_deck(point_df, force_areals=False, show_labels=True, height=360)
        if deck:
            st.pydeck_chart(deck, use_container_width=True)
    render_light_dataframe(display_dataframe(point_df, DISPLAY_COLUMNS), height=360)


def render_units_page(df: pd.DataFrame) -> None:
    st.subheader("Лингвистические единицы")
    all_units = get_all_units(df)
    unit = st.selectbox("Выберите единицу", all_units) if all_units else ""
    if not unit:
        st.info("В данных нет лингвистических единиц.")
        return

    subset = filter_dataframe(df, unit_query=unit)
    left, right = st.columns([3, 1.2], gap="large")
    with left:
        deck, points_df, areals = make_deck(
            subset,
            selected_question="Все вопросы",
            color_mode="Диалектные единицы",
            show_areals=True,
            show_isoglosses=True,
            show_labels=True,
            force_areals=True,
            height=520,
        )
        if deck:
            st.pydeck_chart(deck, use_container_width=True)
    with right:
        st.metric("Пунктов", int(without_city_points(subset)["settlement"].nunique()))
        st.metric("Записей", len(subset))
        render_legend(points_df if "points_df" in locals() else pd.DataFrame(), areals if "areals" in locals() else [])
    render_light_dataframe(display_dataframe(subset, DISPLAY_COLUMNS), height=360)


def parse_float_input(value: str, field_name: str) -> float:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        raise ValueError(f"Поле «{field_name}» обязательно.")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"В поле «{field_name}» должно быть число.") from exc


def render_add_settlement_block(df: pd.DataFrame) -> None:
    st.markdown("### Добавить населённый пункт")
    st.caption("Координаты можно найти автоматически по названию, району и региону.")

    c1, c2, c3 = st.columns(3)
    with c1:
        region = st.text_input("Регион", value="Удмуртская Республика", key="add_region")
        settlement = st.text_input("Населённый пункт", key="add_settlement")
    with c2:
        district = st.text_input("Район", key="add_district")
        settlement_type = st.selectbox("Тип пункта", ["село", "деревня", "посёлок", "станция", "починок"], key="add_type")
    with c3:
        landscape = st.text_input("Ландшафт", value="не указан", key="add_landscape")
        st.write("")
        st.write("")
        if st.button("Найти координаты автоматически"):
            found = geocode_settlement(settlement, district, region)
            if found:
                lat, lon, source = found
                st.session_state["add_latitude_text"] = f"{lat:.6f}"
                st.session_state["add_longitude_text"] = f"{lon:.6f}"
                st.success(f"Координаты найдены: {lat:.6f}, {lon:.6f} ({source}).")
            else:
                st.warning("Координаты не найдены автоматически. Введите их вручную.")

    c4, c5 = st.columns(2)
    with c4:
        latitude_text = st.text_input("Широта", key="add_latitude_text")
    with c5:
        longitude_text = st.text_input("Долгота", key="add_longitude_text")

    catalog = question_catalog(df)
    question_labels: list[str] = []
    question_rows: dict[str, pd.Series] = {}
    for _, row in catalog.iterrows():
        label = f"{row['question_type']} · {row['question_id']}. {row['question']}"
        question_labels.append(label)
        question_rows[label] = row

    selected_question = st.selectbox("Вопрос", question_labels, key="add_question") if question_labels else ""

    u1, u2, u3 = st.columns(3)
    with u1:
        unit_1 = st.text_input("Единица 1", key="add_unit1")
    with u2:
        unit_2 = st.text_input("Единица 2", key="add_unit2")
    with u3:
        unit_3 = st.text_input("Единица 3", key="add_unit3")

    comment = st.text_area("Комментарий", key="add_comment")

    if st.button("Добавить запись"):
        try:
            if not settlement.strip():
                raise ValueError("Укажите населённый пункт.")
            if not district.strip():
                raise ValueError("Укажите район.")
            if not selected_question:
                raise ValueError("Выберите вопрос.")

            lat_text = latitude_text
            lon_text = longitude_text
            if not lat_text.strip() or not lon_text.strip():
                found = geocode_settlement(settlement, district, region)
                if found:
                    lat, lon, _ = found
                else:
                    raise ValueError("Координаты не найдены автоматически. Введите широту и долготу вручную.")
            else:
                lat = parse_float_input(lat_text, "Широта")
                lon = parse_float_input(lon_text, "Долгота")

            qrow = question_rows[selected_question]
            record = {
                "region": region.strip(),
                "district": district.strip(),
                "settlement": settlement.strip(),
                "settlement_type": settlement_type.strip(),
                "latitude": lat,
                "longitude": lon,
                "landscape": landscape.strip() or "не указан",
                "atlas_system": str(qrow["atlas_system"]),
                "question_type": str(qrow["question_type"]),
                "question_id": str(qrow["question_id"]),
                "question": str(qrow["question"]),
                "linguistic_unit_1": unit_1.strip(),
                "linguistic_unit_2": unit_2.strip(),
                "linguistic_unit_3": unit_3.strip(),
                "comment": comment.strip(),
                "source": "",
                "year": "",
            }
            st.session_state.setdefault("added_records", []).append(record)
            st.success("Запись добавлена в текущую сессию. Её можно скачать вместе с таблицей.")
        except ValueError as exc:
            st.error(str(exc))


def render_table_page(df: pd.DataFrame, editor_url: str) -> None:
    st.subheader("Таблица данных")
    if editor_url:
        st.link_button("Открыть таблицу для редактирования", editor_url)
    else:
        st.info("Добавьте REMOTE_TABLE_EDIT_URL в secrets или вставьте ссылку в боковой панели, чтобы показывать кнопку редактирования.")

    issues = validate_dataframe(df)
    st.markdown("### Проверка таблицы")
    if issues:
        st.dataframe(pd.DataFrame(issues), hide_index=True, use_container_width=True)
    else:
        st.success("Ошибок в обязательных полях не найдено.")

    with st.expander("Добавить населённый пункт", expanded=False):
        render_add_settlement_block(df)

    st.markdown("### Текущая таблица")
    st.caption("В отображении убраны столбцы: атлас, год, источник. Номер вопроса показан как 1, 2, 3.")
    visible_df = display_dataframe(df, DISPLAY_COLUMNS)
    render_light_dataframe(visible_df, height=520, max_rows=700)

    st.markdown("### Редактор текущей копии")
    st.caption("Изменения здесь не записываются в Google Sheets автоматически; скачайте CSV после правок.")
    editor_base = display_dataframe(df, DISPLAY_COLUMNS)
    st.data_editor(
        editor_base,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="local_data_editor",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Скачать текущую полную CSV",
            data=to_download_csv(df),
            file_name="dialekt_udmurtii_edited.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "Скачать шаблон таблицы",
            data=read_template_bytes(),
            file_name="dialekt_udmurtii_template.csv",
            mime="text/csv",
        )

    st.markdown("### Обязательная структура")
    schema = pd.DataFrame(
        [
            ["region", "область / край / республика", "да"],
            ["district", "район", "да"],
            ["settlement", "населённый пункт", "да"],
            ["settlement_type", "тип населённого пункта", "желательно"],
            ["latitude, longitude", "координаты WGS84", "для карты"],
            ["landscape", "ландшафт / тип местности", "желательно"],
            ["question_type", "раздел вопроса: ДАРЯ/ЛАРНГ", "да"],
            ["question_id", "номер вопроса 1, 2, 3", "желательно"],
            ["question", "формулировка вопроса", "да"],
            ["linguistic_unit_1..n", "варианты / ответы", "да"],
            ["comment", "комментарий к пункту или карте", "желательно"],
        ],
        columns=["Поле", "Что хранит", "Статус"],
    )
    render_light_dataframe(schema, height=300)


def render_gis_page() -> None:
    st.subheader("ГИС-справочник для проекта")
    st.markdown(
        """
Этот раздел нужен, чтобы связать учебное приложение с географическими информационными системами:
подготовка границ районов, импорт CSV, экспорт GeoJSON, ручная правка изоглосс, публикация веб-слоёв.
"""
    )
    systems = pd.DataFrame(
        [
            ["QGIS", "Подготовка GeoJSON", "бесплатная настольная ГИС"],
            ["ArcGIS", "Редактирование слоёв и публикация", "профессиональная ГИС"],
            ["NextGIS", "Веб-публикация слоёв", "удобно для онлайн-карт"],
            ["MapInfo", "Работа с табличными и векторными слоями", "часто используется в региональных проектах"],
            ["ГИС Панорама", "Топографические данные", "поддержка российских форматов"],
        ],
        columns=["Система", "Назначение", "Комментарий"],
    )
    render_light_dataframe(systems, height=260)
    st.markdown("### Рекомендуемый рабочий процесс")
    st.markdown(
        """
1. В QGIS/ArcGIS/NextGIS подготовьте слой районов, регионов или ландшафтных зон.
2. Проверьте систему координат: для веб-карты нужен WGS84, EPSG:4326.
3. Экспортируйте слой в GeoJSON и опубликуйте файл по HTTPS либо добавьте его в репозиторий.
4. Укажите ссылку в `BOUNDARY_GEOJSON_URL` в secrets, если нужны точные границы вместо встроенных.
5. Табличные диалектные данные храните отдельно: Google Sheets/CSV остаётся главным источником.
"""
    )


def render_help_page() -> None:
    st.subheader("Инструкция пользователя")
    st.markdown(
        """
### Для зрителя
На этой странице сверху расположена карта. Настройте фильтры в боковой панели: регион, район,
конкретный вопрос, поиск единицы или общий поиск. Точки показывают только не городские
населённые пункты. При наведении видны вопрос, единицы, ландшафт и комментарий.

### Для редактора таблицы
Заполняйте Google Sheets по шаблону: одна строка = один населённый пункт + один вопрос.
Для нескольких ответов используйте столбцы `linguistic_unit_1`, `linguistic_unit_2`,
`linguistic_unit_3` или разделяйте варианты точкой с запятой. После редактирования нажмите
**Обновить кэш данных** в боковой панели.

### Для добавления пункта
Ниже карты расположен блок **Добавить населённый пункт**. Если координаты не введены,
приложение попробует найти их автоматически по названию, району и региону.

### Для составителя карт
В демо-данных оставлено 3 вопроса ДАРЯ и 3 вопроса ЛАРНГ. Номера вопросов отображаются как
1, 2, 3. Автоматические ареалы строятся по текущей выборке, в том числе после общего поиска.
"""
    )
    st.markdown("### Допустимые значения `question_type`")
    render_light_dataframe(pd.DataFrame({"question_type": ALLOWED_QUESTION_TYPES}), height=220)
    st.markdown("### Быстрый запуск")
    st.code("pip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")


def main() -> None:
    st.set_page_config(
        page_title="Интерактивная карта русских говоров Удмуртии",
        page_icon="🗺️",
        layout="wide",
    )
    inject_css()

    st.title("Интерактивная карта русских говоров Удмуртии")
    st.caption("Пункты, ареалы, изоглоссы и таблица диалектных данных")

    base_df, source_note, editor_url = load_data_from_sidebar()
    df = get_working_dataframe(base_df)
    filters = sidebar_filters(df)

    st.sidebar.caption(source_note)
    city_count = int(add_unit_display(df)["settlement_type"].str.lower().str.contains("город", na=False).sum())
    st.sidebar.caption(f"Городские строки скрыты на карте: {city_count}")

    st.markdown("## Карта")
    filtered = filter_dataframe(
        df,
        regions=filters["regions"],
        districts=filters["districts"],
        question=filters["selected_question"],
        unit_query=filters["unit_query"],
        text_query=filters["text_query"],
    )
    force_areals = bool(filters["text_query"].strip() or filters["unit_query"].strip())

    left, right = st.columns([3.2, 1.15], gap="large")
    with left:
        deck, points_df, areals = make_deck(
            filtered,
            selected_question=filters["selected_question"],
            color_mode=filters["color_mode"],
            show_areals=filters["show_areals"],
            show_isoglosses=filters["show_isoglosses"],
            show_labels=filters["show_labels"],
            force_areals=force_areals,
        )
        if deck is None:
            st.warning("Нет не городских пунктов с координатами для выбранных фильтров.")
        else:
            st.pydeck_chart(deck, use_container_width=True)

    with right:
        st.caption("Легенда")
        render_legend(points_df if "points_df" in locals() else pd.DataFrame(), areals if "areals" in locals() else [])
        st.metric("Пунктов на карте", int(points_df["settlement"].nunique()) if "points_df" in locals() and not points_df.empty else 0)
        st.metric("Ареалов", len(areals) if "areals" in locals() else 0)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Добавить населённый пункт и вопрос")
    render_add_settlement_block(df)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Таблица")
    if editor_url:
        st.link_button("Открыть таблицу для редактирования", editor_url)

    issues = validate_dataframe(df)
    if issues:
        st.markdown("### Проверка таблицы")
        st.dataframe(pd.DataFrame(issues), hide_index=True, use_container_width=True)

    st.caption("В отображении убраны столбцы: атлас, год, источник. Номер вопроса показан как 1, 2, 3.")
    visible_df = display_dataframe(filtered, DISPLAY_COLUMNS)
    render_light_dataframe(visible_df, height=520, max_rows=700)

    table_dl_1, table_dl_2 = st.columns(2)
    with table_dl_1:
        st.download_button(
            "Скачать выбранные записи CSV",
            data=to_download_csv(filtered),
            file_name="dialekt_selected_records.csv",
            mime="text/csv",
        )
    with table_dl_2:
        st.download_button(
            "Скачать текущую полную CSV",
            data=to_download_csv(df),
            file_name="dialekt_udmurtii_edited.csv",
            mime="text/csv",
        )

    with st.expander("Редактор текущей копии", expanded=False):
        st.caption("Изменения здесь не записываются в Google Sheets автоматически; скачайте CSV после правок.")
        editor_base = display_dataframe(df, DISPLAY_COLUMNS)
        st.data_editor(
            editor_base,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="local_data_editor",
        )
        st.download_button(
            "Скачать шаблон таблицы",
            data=read_template_bytes(),
            file_name="dialekt_udmurtii_template.csv",
            mime="text/csv",
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    render_help_page()


if __name__ == "__main__":
    main()
