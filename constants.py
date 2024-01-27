import os
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    # chroma_db_impl="duckdb+parquet",  This option is deprecated. Chroma 0.4~
    persist_directory="db",
    anonymized_telemetry=False,
)
