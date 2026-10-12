from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4

import duckdb
import fsspec
import pytest
from test_smoke import _prepare_abp_test_parquet, _prepare_test_parquet

from ukam_os_builder._exceptions import ToFlatfileError
from ukam_os_builder.api.settings import (
    OSDownloadSettings,
    OutputSettings,
    PathSettings,
    ProcessingSettings,
    Settings,
    SettingsError,
    load_settings,
)
from ukam_os_builder.os_builder.output import (
    WRITERS,
    AzureWriter,
    LocalWriter,
    S3Writer,
    Writer,
    build_writer,
)


@pytest.fixture
def cloud_settings(tmp_path: Path) -> Settings:
    return Settings(
        paths=PathSettings(
            work_dir=tmp_path,
            downloads_dir=tmp_path / "downloads",
            extracted_dir=tmp_path / "extracted",
            output_dir="s3://test-bucket/output",
        ),
        os_downloads=OSDownloadSettings(package_id="1", version_id="1"),
        processing=ProcessingSettings(num_chunks=2),
        config_path=tmp_path / "config.yaml",
    )


@pytest.fixture
def destination(cloud_settings: Settings) -> Writer:
    output = build_writer(cloud_settings, "ngd")
    filesystem = fsspec.filesystem("memory", skip_instance_cache=True)
    filesystem.store.clear()
    output._filesystem = filesystem
    return output


class RecordingConnection:
    def __init__(self, destination: Writer | None = None) -> None:
        self.statements: list[str] = []
        self.destination = destination

    def execute(self, sql: str) -> None:
        self.statements.append(sql)
        if sql.startswith("COPY") and self.destination:
            name = re.search(r"(_ukam_probe_[a-z0-9]+\.parquet)", sql).group(1)
            self.destination.filesystem.pipe_file(f"{self.destination.root}/{name}", b"probe")


def write_chunks(destination: Writer) -> None:
    for index in range(destination.settings.processing.num_chunks):
        destination.filesystem.pipe_file(
            f"{destination.root}/{destination.chunk_name(index)}", b"parquet"
        )
        destination.record_chunk(index, index)


def test_cloud_completion_and_reuse(destination: Writer) -> None:
    destination.prepare(RecordingConnection(destination), force=False)
    write_chunks(destination)
    destination.finish()
    assert destination.existing_outputs(force=False) == [
        str(destination.target(destination.chunk_name(index))) for index in range(2)
    ]
    assert destination.completed_manifest().chunks[0].rows == 0


def test_partial_cloud_run_requires_overwrite(destination: Writer) -> None:
    destination.filesystem.pipe_file(f"{destination.root}/{destination.chunk_name(0)}", b"partial")
    with pytest.raises(ToFlatfileError, match="without a completed run"):
        destination.existing_outputs(force=False)
    with pytest.raises(ToFlatfileError, match="Not all cloud chunks"):
        destination.finish()
    assert not destination.filesystem.exists(f"{destination.root}/{destination.manifest_name}")


def test_overwrite_removes_all_old_generations_only_for_own_source(
    destination: Writer,
) -> None:
    destination.settings.processing.num_chunks = 10
    write_chunks(destination)
    destination.finish()
    unrelated = [
        f"{destination.root}/abp_for_uk_address_matcher.chunk_001_of_001.parquet",
        f"{destination.root}/other.parquet",
        f"{destination.root}/nested/ngd_for_uk_address_matcher.chunk_001_of_010.parquet",
        "test-bucket/output-neighbor/ngd_for_uk_address_matcher.chunk_001_of_010.parquet",
    ]
    for key in unrelated:
        destination.filesystem.pipe_file(key, b"unrelated")
    destination.settings.processing.num_chunks = 5
    destination.prepare(RecordingConnection(destination), force=True)
    write_chunks(destination)
    destination.finish()
    assert len(destination._owned_files()) == 5
    assert all(destination.filesystem.exists(key) for key in unrelated)


def test_modified_chunk_invalidates_completion(destination: Writer) -> None:
    write_chunks(destination)
    destination.finish()
    destination.filesystem.pipe_file(
        f"{destination.root}/{destination.chunk_name(0)}", b"different-size"
    )
    with pytest.raises(ToFlatfileError, match="incomplete"):
        destination.existing_outputs(force=False)


def test_changed_configuration_invalidates_completion(destination: Writer) -> None:
    write_chunks(destination)
    destination.finish()
    destination.settings.processing.ngd_excluded_stems = []
    with pytest.raises(ToFlatfileError, match="differs"):
        destination.existing_outputs(force=False)


def test_manifest_publication_failure_leaves_no_completion(
    destination: Writer, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chunks(destination)
    original = destination.filesystem.pipe_file

    def fail_publication(key: str, data: bytes) -> None:
        original(key, data)
        raise OSError("upload failed")

    monkeypatch.setattr(destination.filesystem, "pipe_file", fail_publication)
    with pytest.raises(OSError, match="upload failed"):
        destination.finish()
    assert not destination.filesystem.exists(f"{destination.root}/{destination.manifest_name}")


def test_delete_failure_preserves_old_manifest_before_overwrite(
    destination: Writer, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_chunks(destination)
    destination.finish()
    original = destination.filesystem.rm

    def deny_probe_delete(key: str) -> None:
        if "_ukam_probe_" in key:
            raise PermissionError("delete denied")
        original(key)

    monkeypatch.setattr(destination.filesystem, "rm", deny_probe_delete)
    with pytest.raises(PermissionError, match="delete denied"):
        destination.prepare(RecordingConnection(destination), force=True)
    assert destination.completed_manifest() is not None


def test_s3_secret_scopes_and_refreshes_credentials(cloud_settings: Settings) -> None:
    cloud_settings.output = OutputSettings(
        aws={
            "region": "eu-west-2",
            "profile": "development",
            "credential_chain": ["sso"],
        },
    )
    connection = RecordingConnection()
    build_writer(cloud_settings, "ngd").configure_connection(connection)
    assert connection.statements[:4] == ["INSTALL httpfs", "LOAD httpfs", "INSTALL aws", "LOAD aws"]
    assert "TEMPORARY SECRET" in connection.statements[-1]
    assert "SCOPE 's3://test-bucket/output/'" in connection.statements[-1]
    assert "REFRESH auto" in connection.statements[-1]
    assert "PROFILE 'development'" in connection.statements[-1]


def test_azure_normalizes_filesystem_path_and_escapes_scope(cloud_settings: Settings) -> None:
    cloud_settings.paths.output_dir = "az://testaccount.blob.core.windows.net/outputs/ngd's"
    destination = build_writer(cloud_settings, "ngd")
    connection = RecordingConnection()
    destination.configure_connection(connection)
    assert destination.root == "outputs/ngd's"
    assert "ACCOUNT_NAME 'testaccount'" in connection.statements[-1]
    assert "ngd''s/'" in connection.statements[-1]


def test_kms_uses_connection_setting_without_snapshotting_credentials(
    cloud_settings: Settings,
) -> None:
    cloud_settings.output = OutputSettings(aws={"kms_key_id": "test-key"})
    connection = RecordingConnection()
    build_writer(cloud_settings, "ngd").configure_connection(connection)
    assert "SET s3_kms_key_id = 'test-key'" in connection.statements
    assert "KMS_KEY_ID" not in connection.statements[-1]
    assert "REFRESH auto" in connection.statements[-1]


def test_s3_filesystem_mirrors_explicit_chain_and_region(
    cloud_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("aiobotocore")
    cloud_settings.output = OutputSettings(
        aws={
            "region": "eu-west-2",
            "credential_chain": ["process", "env"],
            "kms_key_id": "test-key",
        },
    )
    captured = {}

    def filesystem(protocol, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(fsspec, "filesystem", filesystem)
    _filesystem = build_writer(cloud_settings, "ngd").filesystem
    resolver = captured["session"].get_component("credential_provider")
    assert [provider.METHOD for provider in resolver.providers] == ["custom-process", "env"]
    assert captured["client_kwargs"] == {"region_name": "eu-west-2"}
    assert captured["s3_additional_kwargs"]["SSEKMSKeyId"] == "test-key"


def test_missing_cloud_extra_has_installation_remedy(
    cloud_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args, **kwargs):
        raise ImportError("s3fs missing")

    monkeypatch.setattr(fsspec, "filesystem", unavailable)
    with pytest.raises(SettingsError, match=r"ukam-os-builder\[aws\]"):
        _filesystem = build_writer(cloud_settings, "ngd").filesystem


def test_old_duckdb_rejected_before_install(
    cloud_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(duckdb, "__version__", "1.4.2")
    connection = RecordingConnection()
    with pytest.raises(SettingsError, match="duckdb>=1.4.3"):
        build_writer(cloud_settings, "ngd").configure_connection(connection)
    assert connection.statements == []


def test_local_paths_and_cleanup_are_source_scoped(cloud_settings: Settings) -> None:
    cloud_settings.output = None
    cloud_settings.paths.output_dir = cloud_settings.paths.work_dir / "output"
    destination = build_writer(cloud_settings, "ngd")
    directory = cloud_settings.paths.output_dir
    directory.mkdir()
    owned = directory / destination.chunk_name(0)
    owned.write_bytes(b"parquet")
    other = directory / "abp_for_uk_address_matcher.chunk_001_of_001.parquet"
    other.write_bytes(b"parquet")
    assert destination.existing_outputs(False) == [owned]
    destination.prepare(RecordingConnection(), force=True)
    assert not owned.exists()
    assert other.exists()
    assert isinstance(destination.target(destination.chunk_name(0)), Path)


@pytest.mark.parametrize("change", ["missing", "extra", "corrupt"])
def test_invalid_cloud_output_never_skips(destination: Writer, change: str) -> None:
    write_chunks(destination)
    destination.finish()
    if change == "missing":
        destination.filesystem.rm(f"{destination.root}/{destination.chunk_name(0)}")
    elif change == "extra":
        destination.filesystem.pipe_file(
            f"{destination.root}/ngd_for_uk_address_matcher.chunk_001_of_010.parquet", b"stale"
        )
    else:
        destination.filesystem.pipe_file(
            f"{destination.root}/{destination.manifest_name}", b"not json"
        )
    with pytest.raises(ToFlatfileError, match="--overwrite"):
        destination.existing_outputs(False)


def test_memory_limit_change_does_not_invalidate_completion(destination: Writer) -> None:
    write_chunks(destination)
    destination.finish()
    destination.settings.processing.duckdb_memory_limit = "1GB"
    assert destination.existing_outputs(False)


def test_azure_filesystem_receives_account_not_duckdb_uri(
    cloud_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud_settings.paths.output_dir = "az://testaccount.blob.core.windows.net/outputs/ngd"
    captured = {}

    def filesystem(protocol, **kwargs):
        captured.update(kwargs)
        assert protocol == "az"
        return object()

    monkeypatch.setattr(fsspec, "filesystem", filesystem)
    _filesystem = build_writer(cloud_settings, "ngd").filesystem
    assert captured["account_name"] == "testaccount"
    assert captured["anon"] is False


def test_connection_closes_before_scratch_cleanup(
    cloud_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ukam_os_builder.os_builder.output as output_module

    closed = []

    class Connection:
        def execute(self, sql):
            pass

        def close(self):
            closed.append(bool(list(cloud_settings.paths.work_dir.glob("ukam_duckdb_*"))))

    monkeypatch.setattr(output_module, "create_duckdb_connection", lambda settings: Connection())
    with (
        pytest.raises(RuntimeError, match="write failed"),
        build_writer(cloud_settings, "ngd").connection(),
    ):
        raise RuntimeError("write failed")
    assert closed == [True]
    assert not list(cloud_settings.paths.work_dir.glob("ukam_duckdb_*"))


@pytest.mark.cloud
@pytest.mark.parametrize(
    "provider,uri_variable", [("s3", "UKAM_TEST_S3_URI"), ("azure", "UKAM_TEST_AZURE_URI")]
)
@pytest.mark.parametrize("source", ["ngd", "abp"])
def test_real_cloud_writers_match_local_and_replace_chunk_generation(
    cloud_settings: Settings, provider: str, uri_variable: str, source: str
) -> None:
    from ukam_os_builder.data_sources.abp.transform.runner import transform_to_flatfile
    from ukam_os_builder.data_sources.ngd.to_flatfile import run_flatfile_step

    base_uri = os.environ.get(uri_variable)
    if not base_uri:
        pytest.skip(f"Set {uri_variable} to a dedicated test prefix to enable real cloud I/O")
    temp_settings = cloud_settings
    temp_settings.output = None
    temp_settings.paths.output_dir = temp_settings.paths.work_dir / "output"
    local_output_dir = temp_settings.paths.output_dir
    temp_settings.processing.num_chunks = 1
    temp_settings.processing.ngd_excluded_stems = []
    (temp_settings.paths.extracted_dir / "parquet").mkdir(parents=True, exist_ok=True)
    temp_settings.source.type = source
    writer = run_flatfile_step if source == "ngd" else transform_to_flatfile
    if source == "ngd":
        _prepare_test_parquet(temp_settings)
    else:
        _prepare_abp_test_parquet(temp_settings)
    baseline_paths = writer(temp_settings, force=True)
    with duckdb.connect() as connection:
        baseline = (
            connection.read_parquet([str(path) for path in baseline_paths]).order("ALL").fetchall()
        )
    uri = base_uri.rstrip("/") + f"/ukam-test-{uuid4().hex}-{source}"
    aws = None
    if provider == "s3":
        aws = {
            "region": os.environ.get("AWS_REGION"),
            "profile": os.environ.get("UKAM_TEST_AWS_PROFILE"),
        }
        chain = os.environ.get("UKAM_TEST_AWS_CHAIN")
        if chain:
            aws["credential_chain"] = chain.split(";")
        if os.environ.get("UKAM_TEST_KMS_KEY_ID"):
            aws["kms_key_id"] = os.environ["UKAM_TEST_KMS_KEY_ID"]
    temp_settings.paths.output_dir = uri
    temp_settings.output = OutputSettings(aws=aws)
    destination = build_writer(temp_settings, source)
    try:
        outputs = writer(temp_settings, force=True)
        assert all(isinstance(path, str) and path.startswith(uri + "/") for path in outputs)
        assert writer(temp_settings) == outputs
        temp_settings.processing.num_chunks = 2
        with pytest.raises(ToFlatfileError, match="--overwrite"):
            writer(temp_settings)
        outputs = writer(temp_settings, force=True)
        assert len(outputs) == 2
        assert len(destination._owned_files()) == 2
        assert sum(chunk.rows for chunk in destination.completed_manifest().chunks) == len(baseline)
        with destination.connection() as connection:
            destination.configure_connection(connection)
            assert connection.read_parquet(outputs).order("ALL").fetchall() == baseline
        assert sorted(local_output_dir.glob("*.parquet")) == baseline_paths
    finally:
        if destination.filesystem.exists(destination.root):
            destination.filesystem.rm(destination.root, recursive=True)


def test_s3_metadata_region_matches_duckdb_environment(
    cloud_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    monkeypatch.setenv("AWS_REGION", "eu-west-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    def filesystem(protocol, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(fsspec, "filesystem", filesystem)
    _filesystem = build_writer(cloud_settings, "ngd").filesystem
    assert captured["client_kwargs"] == {"region_name": "eu-west-2"}


@pytest.mark.parametrize(
    "path,expected",
    [
        ("s3://test-bucket/output", S3Writer),
        ("az://testaccount.blob.core.windows.net/outputs/ngd", AzureWriter),
        ("local-output", LocalWriter),
        ("unknown://destination/output", LocalWriter),
    ],
)
def test_factory_selects_writer_from_output_path(
    cloud_settings: Settings, caplog: pytest.LogCaptureFixture, path: str, expected: type[Writer]
) -> None:
    cloud_settings.paths.output_dir = path
    with caplog.at_level("INFO"):
        writer = build_writer(cloud_settings, "ngd")
    assert isinstance(writer, expected)
    if expected is LocalWriter:
        assert "Writing output locally" in caplog.text


def test_writer_is_an_abstract_strategy(cloud_settings: Settings) -> None:
    with pytest.raises(TypeError, match="abstract"):
        Writer(cloud_settings, "ngd")


def test_unknown_base_falls_back_locally_relative_to_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        'paths: {work_dir: "other://location/base"}\n'
        'os_downloads: {package_id: "1", version_id: "1"}\n',
        encoding="utf-8",
    )
    settings = load_settings(config, load_env=False)
    with caplog.at_level("INFO"):
        writer = build_writer(settings, "ngd")
    assert isinstance(writer, LocalWriter)
    assert writer.directory == tmp_path / "other:/location/base/output"
    assert settings.paths.work_dir == tmp_path / "data"
    assert "Writing output locally:" in caplog.text


def test_factory_accepts_an_additional_strategy(
    cloud_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CustomWriter(LocalWriter):
        pass

    monkeypatch.setitem(WRITERS, "custom", CustomWriter)
    cloud_settings.paths.output_dir = "custom://location/final"
    assert isinstance(build_writer(cloud_settings, "ngd"), CustomWriter)
