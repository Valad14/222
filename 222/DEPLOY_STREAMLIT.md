# Деплой в Streamlit Community Cloud

1. Загрузите папку `222` в репозиторий GitHub.
2. В Streamlit Community Cloud выберите файл запуска `222/streamlit_app.py`.
3. При необходимости добавьте secrets:

```toml
DATA_CSV_URL = "https://.../published.csv"
REMOTE_TABLE_EDIT_URL = "https://..."
BOUNDARY_GEOJSON_URL = "https://.../boundaries.geojson"
```

`DATA_CSV_URL` должен вести на CSV-файл, например опубликованную таблицу Google Sheets.
