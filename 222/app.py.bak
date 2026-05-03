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
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        .legend-card {
            border: 1px solid #dbe3ef;
            border-radius: 12px;
            padding: 8px 9px;
            background: #ffffff;
        }
        .legend-summary {
            margin-bottom: 6px;
            font-size: 12px;
            font-weight: 700;
            color: #0f172a;
        }
        .legend-list {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .legend-row {
            display: grid;
            grid-template-columns: 11px minmax(0, 1fr) auto;
            align-items: center;
            column-gap: 5px;
            min-height: 17px;
            font-size: 11.5px;
            line-height: 1.12;
            color: #1f2937;
        }
        .legend-swatch {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, .35);
        }
        .legend-label {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .legend-count {
            color: #64748b;
            font-variant-numeric: tabular-nums;
        }
        .legend-note {
            margin-top: 7px;
            font-size: 11px;
            line-height: 1.2;
            color: #64748b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_light_dataframe(df: pd.DataFrame, height: int = 430, max_rows: int = 500) -> None:
    if df.empty:
        st.info("В таблице нет записей для выбранных фильтров.")
        return
    shown = df.head(max_rows).copy()
    if len(df) > max_rows:
        st.caption(f"Показано {max_rows} из {len(df)} строк. Сузьте фильтры, чтобы увидеть меньшую выборку.")
    st.dataframe(shown, hide_index=True, use_container_width=True, height=height)


def render_editable_dataframe(
    df: pd.DataFrame,
    key: str,
    height: int = 520,
    max_rows: int = 700,
) -> pd.DataFrame:
    """Показывает интерактивную редактируемую таблицу и возвращает видимые правки."""
    if df.empty:
        st.info("В таблице нет записей для выбранных фильтров.")
        return df.copy()

    shown = df.head(max_rows).copy()
    if len(df) > max_rows:
        st.caption(f"Показано {max_rows} из {len(df)} строк. Сузьте фильтры, чтобы редактировать меньшую выборку.")

    return st.data_editor(
        shown,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        height=height,
        key=key,
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
            return cached_url_csv(url), "Удалённая таблица CSV / Google Sheets", editor_url
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


def filters_are_default(filters: dict, df: pd.DataFrame) -> bool:
    """True только для стартового состояния карты: фильтры фактически не заданы."""
    map_df = without_city_points(add_unit_display(df))
    all_regions = sorted([x for x in map_df["region"].dropna().unique() if x])
    selected_regions = sorted([x for x in filters.get("regions", []) if x])

    return (
        selected_regions == all_regions
        and not filters.get("districts")
        and filters.get("selected_question") == "Все вопросы"
        and not str(filters.get("unit_query", "")).strip()
        and not str(filters.get("text_query", "")).strip()
        and filters.get("color_mode") == "Диалектные единицы"
    )


def get_boundaries() -> dict:
    url = _secret("BOUNDARY_GEOJSON_URL")
    if url:
        try:
            return cached_geojson_url(url)
        except Exception as exc:
            st.warning(f"GeoJSON-слой из URL не загружен, показаны встроенные границы: {exc}")
    return DEFAULT_REGION_BOUNDARIES


def make_deck(
    df: pd.DataFrame,
    selected_question: str = "Все вопросы",
    color_mode: str = "Диалектные единицы",
    show_areals: bool = True,
    show_isoglosses: bool = True,
    show_labels: bool = False,
    force_areals: bool = False,
    single_color: bool = False,
    height: int = 640,
) -> tuple[pdk.Deck | None, pd.DataFrame, list[dict]]:
    df = add_unit_display(df)
    map_df = without_city_points(df)
    points_df = aggregate_points(map_df)
    if points_df.empty:
        return None, points_df, []

    points_df = add_point_visuals(points_df, color_mode, single_color=single_color)
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
            "html": "{tooltip}<br>Цвет: {color_label}",
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


def _deck_color_to_hex(color: object, fallback: str = "#2563eb") -> str:
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            red, green, blue = [max(0, min(255, int(value))) for value in color[:3]]
            return f"#{red:02x}{green:02x}{blue:02x}"
        except Exception:
            return fallback
    return fallback


def render_legend(points_df: pd.DataFrame, areals: list[dict]) -> None:
    if points_df.empty:
        st.info("Легенда появится, когда на карте будут точки.")
        return

    labels = (
        points_df.groupby("color_label", dropna=False)
        .agg(points=("settlement", "nunique"), legend_color=("color", "first"))
        .reset_index()
        .sort_values(["points", "color_label"], ascending=[False, True])
    )

    summary = f"{len(labels)} категорий"
    if areals:
        summary += f" · {len(areals)} ареалов"

    legend_html = [
        f"<div class='legend-card'><div class='legend-summary'>{html.escape(summary)}</div><div class='legend-list'>"
    ]
    for _, row in labels.iterrows():
        label = str(row["color_label"])
        color = _deck_color_to_hex(row.get("legend_color"))
        legend_html.append(
            "<div class='legend-row'>"
            f"<span class='legend-swatch' style='background:{color}'></span>"
            f"<span class='legend-label' title='{html.escape(label)}'>{html.escape(label)}</span>"
            f"<span class='legend-count'>{int(row['points'])}</span>"
            "</div>"
        )

    legend_html.append("</div>")
    if areals:
        legend_html.append("<div class='legend-note'>Ареалы построены по текущей выборке, включая общий поиск.</div>")
    legend_html.append("</div>")
    st.markdown("\n".join(legend_html), unsafe_allow_html=True)


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
    st.caption("Координаты можно найти автоматически по названию, району и региону. Запись добавляется в текущую сессию.")

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

            if not latitude_text.strip() or not longitude_text.strip():
                found = geocode_settlement(settlement, district, region)
                if found:
                    lat, lon, _ = found
                else:
                    raise ValueError("Координаты не найдены автоматически. Введите широту и долготу вручную.")
            else:
                lat = parse_float_input(latitude_text, "Широта")
                lon = parse_float_input(longitude_text, "Долгота")

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


def render_table_section(df: pd.DataFrame, filtered: pd.DataFrame, editor_url: str) -> None:
    st.markdown("## Таблица")
    if editor_url:
        st.link_button("Открыть таблицу для редактирования", editor_url)

    issues = validate_dataframe(df)
    if issues:
        with st.expander("Проверка таблицы", expanded=True):
            st.dataframe(pd.DataFrame(issues), hide_index=True, use_container_width=True)
    else:
        st.success("Ошибок в обязательных полях не найдено.")

    st.caption("В отображении убраны столбцы: атлас, год, источник. Номер вопроса показан как 1, 2, 3.")
    visible_df = display_dataframe(filtered, DISPLAY_COLUMNS)
    edited_visible_df = render_editable_dataframe(visible_df, key="filtered_data_editor", height=520, max_rows=700)
    st.caption("Таблица интерактивная: ячейки можно менять, строки можно добавлять или удалять. Правки остаются в текущей сессии; для постоянного сохранения скачайте CSV.")

    table_dl_1, table_dl_2, table_dl_3 = st.columns(3)
    with table_dl_1:
        st.download_button(
            "Скачать отредактированную видимую таблицу CSV",
            data=edited_visible_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="dialekt_selected_visible_edited.csv",
            mime="text/csv",
        )
    with table_dl_2:
        st.download_button(
            "Скачать выбранные записи CSV",
            data=to_download_csv(filtered),
            file_name="dialekt_selected_records.csv",
            mime="text/csv",
        )
    with table_dl_3:
        st.download_button(
            "Скачать текущую полную CSV",
            data=to_download_csv(df),
            file_name="dialekt_udmurtii_edited.csv",
            mime="text/csv",
        )

    with st.expander("Шаблон и полная копия", expanded=False):
        st.download_button(
            "Скачать шаблон таблицы",
            data=read_template_bytes(),
            file_name="dialekt_udmurtii_template.csv",
            mime="text/csv",
        )
        st.markdown("### Полный редактор текущей копии")
        full_editor = display_dataframe(df, DISPLAY_COLUMNS)
        st.data_editor(
            full_editor,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="local_data_editor",
        )


def render_help_page() -> None:
    st.markdown("## Инструкция пользователя")
    st.markdown(
        """
        **Карта.** Настройте фильтры в боковой панели: регион, район, конкретный вопрос, поиск единицы или общий поиск.
        Точки показывают только не городские населённые пункты. При наведении видны пункт, количество записей, единицы и ландшафт.

        **Легенда.** Легенда находится справа от карты и раскрыта сразу. Она компактная и показывает все категории текущей раскраски.

        **Таблица.** Основная таблица редактируется прямо на странице. Изменения в ней не записываются автоматически в Google Sheets,
        поэтому после правок скачайте CSV и загрузите его в источник данных.

        **Добавление пункта.** Новый населённый пункт добавляется в текущую сессию. Координаты можно ввести вручную или найти автоматически.
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
            single_color=filters_are_default(filters, df),
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

    st.markdown("---")
    st.markdown("## Добавить населённый пункт и вопрос")
    render_add_settlement_block(df)

    st.markdown("---")
    render_table_section(df, filtered, editor_url)

    st.markdown("---")
    render_help_page()


if __name__ == "__main__":
    main()
