# Деплой в Streamlit Community Cloud

1. Загрузите папку проекта в GitHub.
2. В Streamlit Community Cloud выберите репозиторий и ветку `main`.
3. Main file path: `2/streamlit_app.py` или `streamlit_app.py`, если папка `2` является корнем репозитория.
4. Python: 3.11 или 3.12.
5. При необходимости добавьте secrets:

```toml
DATA_CSV_URL = "https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=0"
REMOTE_TABLE_EDIT_URL = "https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit"
BOUNDARY_GEOJSON_URL = ""
```
