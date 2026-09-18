"""Deterministic transaction and concurrent schema-initialization regressions."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from menagerie import ledger


def _insert_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Insert a minimal row into a test ledger.

    Parameters
    ----------
    conn:
        Initialized test connection.
    run_id:
        Unique run and model identifier.
    """

    conn.execute(
        """
        INSERT INTO verification_runs(
            run_id, stable_id, recipe_revision_sha256, name, zoo, scope, status,
            torchlens_version, torch_version, python_version, device_requested,
            started_at, finished_at, duration_sec, forward_pass
        ) VALUES (?, ?, 'recipe', 'Toy', 'test', 'forward', 'passed',
                  'test', 'test', 'test', 'cpu', '2026-09-17', '2026-09-17', 1.0, 1)
        """,
        (run_id, run_id),
    )


@pytest.mark.parametrize("view", ["current_verification", "current_verification_real"])
def test_view_replacement_is_atomic_across_connections(tmp_path: Path, view: str) -> None:
    """Pause after DROP: readers retain both views and another writer cannot enter."""

    path = tmp_path / "verification.db"
    with closing(ledger.connect(path)) as conn:
        _insert_run(conn, "original")
    dropped, resume, contender_started = Event(), Event(), Event()

    def pause_before_create(sql: str) -> None:
        """Expose the DROP/CREATE window for execute and executescript alike.

        Parameters
        ----------
        sql:
            Statement SQLite is about to execute.
        """

        if sql.lstrip().startswith(f"CREATE VIEW {view} AS"):
            dropped.set()
            resume.wait(10)

    def first_writer() -> None:
        """Initialize using the connection that pauses in the schema window."""

        with closing(sqlite3.connect(path, isolation_level=None)) as conn:
            conn.row_factory = sqlite3.Row
            ledger.configure_connection(conn)
            conn.set_trace_callback(pause_before_create)
            ledger.initialize(conn)
            assert resume.is_set(), "test did not release the initialization transaction"
            _insert_run(conn, "first")

    def second_writer() -> None:
        """Initialize and append through an independent ordinary connection."""

        contender_started.set()
        with closing(ledger.connect(path)) as conn:
            _insert_run(conn, "second")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_writer)
        try:
            assert dropped.wait(10), "writer never reached the view replacement"
            with closing(sqlite3.connect(path, isolation_level=None, timeout=0)) as reader:
                for name in ("current_verification", "current_verification_real"):
                    assert reader.execute(f"SELECT run_id FROM {name}").fetchall() == [
                        ("original",)
                    ]
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    reader.execute("BEGIN IMMEDIATE")
            second = executor.submit(second_writer)
            assert contender_started.wait(10)
        finally:
            resume.set()
        first.result(timeout=10)
        second.result(timeout=10)

    with closing(sqlite3.connect(path)) as conn:
        for name in ("verification_runs", "current_verification", "current_verification_real"):
            assert conn.execute(f"SELECT run_id FROM {name} ORDER BY run_id").fetchall() == [
                ("first",),
                ("original",),
                ("second",),
            ]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM verification_runs")


@pytest.mark.parametrize("existing", [False, True])
def test_failed_initialization_rolls_back_all_schema_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    """A late failure leaves either the original schema or an empty database intact."""

    path = tmp_path / "verification.db"
    if existing:
        with closing(ledger.connect(path)) as conn:
            _insert_run(conn, "original")
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        before = [tuple(row) for row in conn.execute("SELECT * FROM sqlite_master ORDER BY name")]

        def fail_after_changes(connection: sqlite3.Connection) -> None:
            """Fail after replacing a view and adding a column within the transaction.

            Parameters
            ----------
            connection:
                Initializing test connection.
            """

            connection.execute("DROP VIEW current_verification")
            connection.execute("ALTER TABLE verification_runs ADD COLUMN rollback_probe TEXT")
            raise RuntimeError("injected schema failure")

        monkeypatch.setattr(ledger, "_create_current_verification_real_view", fail_after_changes)
        with pytest.raises(RuntimeError, match="injected schema failure"):
            ledger.initialize(conn)
        assert not conn.in_transaction
        assert [
            tuple(row) for row in conn.execute("SELECT * FROM sqlite_master ORDER BY name")
        ] == (before)
        if existing:
            assert conn.execute("SELECT run_id FROM current_verification").fetchone()[0] == (
                "original"
            )


def test_initialization_does_not_commit_callers_transaction(tmp_path: Path) -> None:
    """DDL execution must not trigger executescript's implicit commit of caller data."""

    path = tmp_path / "verification.db"
    with closing(ledger.connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _insert_run(conn, "pending")
        ledger.initialize(conn)
        assert conn.in_transaction
        with closing(sqlite3.connect(path)) as reader:
            assert reader.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0] == 0
        conn.rollback()
        assert conn.execute("SELECT COUNT(*) FROM verification_runs").fetchone()[0] == 0


def test_legacy_status_rebuild_rolls_back_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure restores legacy rows, constraints, indexes, triggers, and both views."""

    with closing(ledger.connect(tmp_path / "current.db")) as current:
        statements = [
            row[0]
            for row in current.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
                "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"
            )
        ]
    with closing(sqlite3.connect(tmp_path / "legacy.db", isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        for statement in statements:
            conn.execute(statement.replace("'killed'", "'legacy_status'"))
        _insert_run(conn, "original")
        before = [tuple(row) for row in conn.execute("SELECT * FROM sqlite_master ORDER BY name")]

        def fail_after_rebuild(connection: sqlite3.Connection) -> None:
            """Confirm the migrated state before injecting a late failure.

            Parameters
            ----------
            connection:
                Initializing test connection.
            """

            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'verification_runs'"
            ).fetchone()[0]
            assert "'killed'" in schema
            assert connection.execute("SELECT run_id FROM verification_runs").fetchone()[0] == (
                "original"
            )
            raise RuntimeError("injected schema failure")

        with monkeypatch.context() as patch:
            patch.setattr(ledger, "_create_current_verification_real_view", fail_after_rebuild)
            with pytest.raises(RuntimeError, match="injected schema failure"):
                ledger.initialize(conn)
        assert not conn.in_transaction
        assert [
            tuple(row) for row in conn.execute("SELECT * FROM sqlite_master ORDER BY name")
        ] == (before)
        ledger.initialize(conn)
        for name in ("verification_runs", "current_verification", "current_verification_real"):
            assert conn.execute(f"SELECT run_id FROM {name}").fetchone()[0] == "original"
        for statement in (
            "DELETE FROM verification_runs",
            "UPDATE verification_runs SET status = 'failed'",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)


def test_failed_initialization_preserves_callers_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rolling back a schema savepoint preserves the caller's pending append."""

    with closing(ledger.connect(tmp_path / "verification.db")) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _insert_run(conn, "pending")

        def fail(connection: sqlite3.Connection) -> None:
            """Inject a failure after the first view was recreated.

            Parameters
            ----------
            connection:
                Initializing test connection.
            """

            connection.execute("DROP VIEW current_verification")
            raise RuntimeError("injected schema failure")

        monkeypatch.setattr(ledger, "_create_current_verification_real_view", fail)
        with pytest.raises(RuntimeError, match="injected schema failure"):
            ledger.initialize(conn)
        assert conn.in_transaction
        assert conn.execute("SELECT run_id FROM current_verification").fetchone()[0] == "pending"
        conn.commit()


def test_failed_connect_closes_its_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed initializer does not leak the newly opened SQLite connection."""

    opened: list[sqlite3.Connection] = []

    def fail(connection: sqlite3.Connection) -> None:
        """Remember the connection before failing initialization.

        Parameters
        ----------
        connection:
            Newly opened connection.
        """

        opened.append(connection)
        raise RuntimeError("injected schema failure")

    monkeypatch.setattr(ledger, "initialize", fail)
    with pytest.raises(RuntimeError, match="injected schema failure"):
        ledger.connect(tmp_path / "verification.db")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
