from __future__ import annotations

from pathlib import Path

import pytest

from ukam_os_builder.api.settings import (
    OSDownloadSettings,
    PathSettings,
    ProcessingSettings,
    Settings,
)


def _test_settings(tmp_path: Path, *, num_chunks: int) -> Settings:
    extracted_dir = tmp_path / "extracted"
    parquet_dir = extracted_dir / "parquet"
    output_dir = tmp_path / "output"

    extracted_dir.mkdir()
    parquet_dir.mkdir()
    output_dir.mkdir()

    return Settings(
        paths=PathSettings(
            work_dir=tmp_path,
            downloads_dir=tmp_path / "downloads",
            extracted_dir=extracted_dir,
            output_dir=output_dir,
        ),
        os_downloads=OSDownloadSettings(
            package_id="test",
            version_id="test",
            api_key="test",
            api_secret="test",
        ),
        processing=ProcessingSettings(
            parquet_compression="zstd",
            parquet_compression_level=1,
            num_chunks=num_chunks,
            ngd_excluded_stems=[],
        ),
        config_path=tmp_path / "config.yaml",
    )


@pytest.fixture
def temp_settings(tmp_path: Path) -> Settings:
    """Create standard test settings backed by pytest's temporary directory."""
    return _test_settings(tmp_path, num_chunks=1)


@pytest.fixture
def temp_settings_chunked(tmp_path: Path) -> Settings:
    """Create standard test settings with two output chunks."""
    return _test_settings(tmp_path, num_chunks=2)
