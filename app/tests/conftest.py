from typing import AsyncIterator

import httpx
import pytest_asyncio

from app.tests.client_helpers import create_app_client


@pytest_asyncio.fixture()
async def fresh_client(tmp_path, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    # Isolate storage into a temp directory for this test
    monkeypatch.setenv("ALAYALITE_DATA_DIR", str(tmp_path))
    # Set RocksDB directory to tmp_path for test isolation
    rocksdb_dir = str(tmp_path / "RocksDB")
    monkeypatch.setenv("ALAYALITE_ROCKSDB_DIR", rocksdb_dir)
    async with create_app_client() as client:
        yield client
