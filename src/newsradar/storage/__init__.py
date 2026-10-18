"""Persistence. SQLite is the default backend; see :mod:`newsradar.storage.base`."""

from newsradar.storage.base import Repository
from newsradar.storage.sqlite import SQLiteRepository, sqlite_path_from_url


def open_repository(url: str) -> Repository:
    """Open the repository described by a storage URL (currently ``sqlite:///...``)."""
    return SQLiteRepository(sqlite_path_from_url(url))


__all__ = ["Repository", "SQLiteRepository", "open_repository"]
