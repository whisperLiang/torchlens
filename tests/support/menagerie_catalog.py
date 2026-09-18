"""Session-private catalog integration fixtures, never the developer's live catalog."""

from __future__ import annotations

import shutil

import pytest
import torch

from menagerie import catalog


@pytest.fixture(scope="session")
def menagerie_rows(tmp_path_factory: pytest.TempPathFactory) -> tuple[catalog.CatalogRow, ...]:
    """Build and round-trip all current rows using a private stable-ID ledger.

    Parameters
    ----------
    tmp_path_factory:
        Pytest's session-private temporary-directory factory.

    Returns
    -------
    tuple[catalog.CatalogRow, ...]
        Real catalog rows loaded from a private SQLite artifact.
    """

    catalog_root = tmp_path_factory.mktemp("menagerie-catalog")
    stable_ids = catalog_root / "stable_ids.jsonl"
    shutil.copyfile(catalog.STABLE_IDS_JSONL, stable_ids)
    with pytest.MonkeyPatch.context() as patch, torch.random.fork_rng(devices=[]):
        patch.setattr(catalog, "STABLE_IDS_JSONL", stable_ids)
        rows = catalog.build_canonical_rows()
    db_path = catalog_root / "catalog.db"
    catalog.write_catalog(rows, canonical_tsv=catalog_root / "catalog.tsv", db_path=db_path)
    loaded = tuple(catalog.load_rows(db_path=db_path))
    assert loaded == tuple(rows)
    return loaded
