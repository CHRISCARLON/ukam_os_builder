from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import duckdb
import pytest
import yaml

from ukam_os_builder.api.settings import (
    OSDownloadSettings,
    PathSettings,
    ProcessingSettings,
    Settings,
    create_duckdb_connection,
)
from ukam_os_builder.data_sources.abp.transform.runner import transform_to_flatfile
from ukam_os_builder.data_sources.ngd.to_flatfile import run_flatfile_step
from ukam_os_builder.os_builder.extract import convert_csv_to_parquet

# Path to sample data (real NGD data from a small area)
SAMPLE_DATA_DIR = Path(__file__).parent / "data"


def _prepare_abp_test_parquet(settings: Settings) -> tuple[int, int]:
    settings.paths.parquet_dir = settings.paths.extracted_dir / "parquet"
    directory = settings.paths.parquet_dir / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    schema_path = (
        Path(__file__).parents[1] / "ukam_os_builder/data_sources/abp/schemas/abp_schema.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text())
    tables = [
        "blpu",
        "lpi",
        "street_descriptor",
        "organisation",
        "delivery_point",
        "classification",
    ]
    with duckdb.connect() as con:
        for table in tables:
            columns = ", ".join(
                f'"{name}" {column["type"]}' for name, column in schema[table]["columns"].items()
            )
            con.execute(f"CREATE TABLE {table} ({columns})")
        parent = con.execute(
            "SELECT MIN(range) FROM range(100, 200) WHERE hash(range) % 2 = 0"
        ).fetchone()[0]
        child = con.execute(
            "SELECT MIN(range) FROM range(100, 200) WHERE hash(range) % 2 = 1"
        ).fetchone()[0]
        con.execute(
            "INSERT INTO blpu (uprn, logical_status, parent_uprn, postcode_locator, country) VALUES (?, 1, NULL, 'SW1A 1AA', 'E'), (?, 1, ?, 'SW1A 1AA', 'E')",
            [parent, child, parent],
        )
        con.execute(
            "INSERT INTO street_descriptor (usrn, street_description, town_name, language) VALUES (1, 'TEST ROAD', 'LONDON', 'ENG')"
        )
        con.execute(
            "INSERT INTO lpi (uprn, lpi_key, logical_status, language, usrn, pao_start_number) VALUES (?, 'parent', 1, 'ENG', 1, 1), (?, 'child', 1, 'ENG', 1, 2)",
            [parent, child],
        )
        for table in tables:
            con.execute(
                f"COPY {table} TO '{(directory / (table + '.parquet')).as_posix()}' (FORMAT PARQUET)"
            )
    return parent, child


def test_abp_chunks_preserve_rows_and_cross_chunk_parent(temp_settings: Settings) -> None:
    parent, child = _prepare_abp_test_parquet(temp_settings)
    baseline_files = transform_to_flatfile(temp_settings, force=True)
    with duckdb.connect() as con:
        baseline = (
            con.read_parquet([str(path) for path in baseline_files])
            .order("unique_id, address_concat")
            .fetchall()
        )
        schema = con.read_parquet([str(path) for path in baseline_files]).columns
    assert baseline
    temp_settings.processing.num_chunks = 2
    chunk_files = transform_to_flatfile(temp_settings, force=True)
    assert len(chunk_files) == 2
    assert all(isinstance(path, Path) for path in chunk_files)
    with duckdb.connect() as con:
        relation = con.read_parquet([str(path) for path in chunk_files])
        assert relation.columns == schema
        assert relation.order("unique_id, address_concat").fetchall() == baseline
        assert (
            relation.filter(f"unique_id = '{parent}'").project("hierarchy_level").fetchone()[0]
            == "P"
        )
        assert relation.filter(f"unique_id = '{child}'").count("*").fetchone()[0] > 0
    assert not list(temp_settings.paths.work_dir.glob("ukam_duckdb_*"))


def test_flatfile_cleanup_belongs_to_destination() -> None:
    from ukam_os_builder.pipeline import _ABP_PIPELINE, _NGD_PIPELINE

    for definition in [_ABP_PIPELINE, _NGD_PIPELINE]:
        assert "output_dir" not in definition.clean_patterns
        assert definition.step_outputs["flatfile"] == []


@pytest.fixture
def temp_settings() -> Generator[Settings, None, None]:
    """Create settings pointing to a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create directories
        extracted_dir = tmpdir_path / "extracted"
        parquet_dir = extracted_dir / "parquet"
        output_dir = tmpdir_path / "output"

        extracted_dir.mkdir()
        parquet_dir.mkdir()
        output_dir.mkdir()

        paths = PathSettings(
            work_dir=tmpdir_path,
            downloads_dir=tmpdir_path / "downloads",
            extracted_dir=extracted_dir,
            output_dir=output_dir,
        )

        # Dummy OS download settings (not used in smoke test)
        os_downloads = OSDownloadSettings(
            package_id="test",
            version_id="test",
            api_key="test",  # pragma: allowlist secret
            api_secret="test",  # pragma: allowlist secret
        )

        processing = ProcessingSettings(
            parquet_compression="zstd",
            parquet_compression_level=1,  # Faster for tests
            num_chunks=1,
            ngd_excluded_stems=[],
        )

        settings = Settings(
            paths=paths,
            os_downloads=os_downloads,
            processing=processing,
            config_path=tmpdir_path / "config.yaml",
        )

        yield settings


@pytest.fixture
def temp_settings_chunked() -> Generator[Settings, None, None]:
    """Create settings with num_chunks=2 for chunking tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        extracted_dir = tmpdir_path / "extracted"
        parquet_dir = extracted_dir / "parquet"
        output_dir = tmpdir_path / "output"

        extracted_dir.mkdir()
        parquet_dir.mkdir()
        output_dir.mkdir()

        paths = PathSettings(
            work_dir=tmpdir_path,
            downloads_dir=tmpdir_path / "downloads",
            extracted_dir=extracted_dir,
            output_dir=output_dir,
        )

        os_downloads = OSDownloadSettings(
            package_id="test",
            version_id="test",
            api_key="test",  # pragma: allowlist secret
            api_secret="test",  # pragma: allowlist secret
        )

        processing = ProcessingSettings(
            parquet_compression="zstd",
            parquet_compression_level=1,
            num_chunks=2,
            ngd_excluded_stems=[],
        )

        settings = Settings(
            paths=paths,
            os_downloads=os_downloads,
            processing=processing,
            config_path=tmpdir_path / "config.yaml",
        )

        yield settings


def _prepare_test_parquet(settings: Settings) -> None:
    """Convert sample CSV files to parquet in the test directory."""
    parquet_dir = settings.paths.extracted_dir / "parquet"

    # Map sample files to expected parquet names (use actual file names)
    sample_files = [
        "add_gb_builtaddress.csv",
        "add_gb_builtaddress_altadd.csv",
        "add_gb_historicaddress.csv",
        "add_gb_royalmailaddress.csv",
        "add_gb_prebuildaddress.csv",
    ]

    for csv_name in sample_files:
        csv_path = SAMPLE_DATA_DIR / csv_name
        if csv_path.exists():
            parquet_name = csv_name.replace(".csv", ".parquet")
            parquet_path = parquet_dir / parquet_name
            convert_csv_to_parquet(csv_path, parquet_path, force=True)


def test_settings_creation(temp_settings: Settings) -> None:
    """Test that settings can be created with test values."""
    assert temp_settings.paths.work_dir.exists()
    assert temp_settings.processing.num_chunks == 1


def test_duckdb_connection(temp_settings: Settings) -> None:
    """Test that DuckDB connection can be created."""
    con = create_duckdb_connection(temp_settings)
    result = con.execute("SELECT 1 AS test").fetchone()
    assert result == (1,)
    con.close()


def test_duckdb_memory_limit_is_logged_once(
    temp_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that repeated configured connections do not repeat the memory log."""
    temp_settings.processing.duckdb_memory_limit = "16GB"
    caplog.clear()

    with caplog.at_level("INFO", logger="ukam_os_builder.api.settings"):
        connections = [create_duckdb_connection(temp_settings) for _ in range(2)]

    for connection in connections:
        connection.close()

    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("Set DuckDB memory limit to 16GB") == 1


def test_flatfile_single_chunk(temp_settings: Settings) -> None:
    """Test flatfile generation with single chunk output."""
    # Prepare test data
    _prepare_test_parquet(temp_settings)
    temp_settings.processing.sort_output_by_postcode = True

    # Run flatfile step
    output_files = run_flatfile_step(temp_settings, force=True)

    # Verify output
    assert len(output_files) == 1
    assert output_files[0].name == "ngd_for_uk_address_matcher.chunk_001_of_001.parquet"
    assert output_files[0].exists()

    # Verify content
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT COUNT(*) as cnt FROM read_parquet('{output_files[0].as_posix()}')
    """).fetchone()
    assert result[0] > 0, "Output should contain records"

    compression = con.execute(
        "SELECT DISTINCT compression FROM parquet_metadata(?)",
        [output_files[0].as_posix()],
    ).fetchall()
    assert compression == [("ZSTD",)]

    ordered_rows = con.execute(f"""
        SELECT postcode, unique_id FROM read_parquet('{output_files[0].as_posix()}')
    """).fetchall()
    assert ordered_rows == sorted(
        ordered_rows,
        key=lambda row: (row[0] is None, row[0] or "", row[1]),
    )

    # Check columns exist
    schema = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{output_files[0].as_posix()}')
    """).fetchall()
    column_names = [row[0] for row in schema]

    expected_columns = [
        "unique_id",
        "address_concat",
        "postcode",
        "filename",
        "classificationcode",
        "parentuprn",
        "lowertierlocalauthoritygsscode",
        "floorlevel",
    ]
    for col in expected_columns:
        assert col in column_names, f"Column {col} should exist in output"

    historic_count = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{output_files[0].as_posix()}')
        WHERE filename = 'add_gb_historicaddress.parquet'
    """).fetchone()[0]
    assert historic_count > 0, "Historic Address records should be processed when included"

    con.close()


def test_flatfile_multi_chunk(temp_settings_chunked: Settings) -> None:
    """Test flatfile generation with multiple chunk output."""
    # Prepare test data
    _prepare_test_parquet(temp_settings_chunked)

    # Run flatfile step
    output_files = run_flatfile_step(temp_settings_chunked, force=True)

    # Verify we have 2 chunks (or fewer if not enough data)
    assert len(output_files) >= 1
    assert len(output_files) <= 2

    for f in output_files:
        assert f.exists()
        assert "chunk_" in f.name


def test_flatfile_idempotent(temp_settings: Settings) -> None:
    """Test that flatfile step is idempotent (safe to re-run)."""
    # Prepare test data
    _prepare_test_parquet(temp_settings)

    # Run twice
    output1 = run_flatfile_step(temp_settings, force=True)
    output2 = run_flatfile_step(temp_settings, force=False)  # Should skip

    assert len(output1) == len(output2)
    assert output1[0].name == output2[0].name


def test_deduplication(temp_settings: Settings) -> None:
    """Test that deduplication removes duplicate UPRN+address combinations."""
    # Prepare test data
    _prepare_test_parquet(temp_settings)

    # Run flatfile step
    output_files = run_flatfile_step(temp_settings, force=True)

    # Verify no exact duplicates
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT unique_id, address_concat, COUNT(*) as cnt
        FROM read_parquet('{output_files[0].as_posix()}')
        GROUP BY unique_id, address_concat
        HAVING COUNT(*) > 1
    """).fetchall()

    assert len(result) == 0, f"Should have no duplicate UPRN+address: {result}"
    con.close()
