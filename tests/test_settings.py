from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from ukam_os_builder.api.settings import SettingsError, load_settings, resolve_output_path


def _write_config(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n")


@pytest.mark.parametrize(
    "base,expected",
    [
        ("s3://test-bucket", "s3://test-bucket/output"),
        ("s3://test-bucket/canonical/", "s3://test-bucket/canonical/output"),
        (
            "az://testaccount.blob.core.windows.net/outputs",
            "az://testaccount.blob.core.windows.net/outputs/output",
        ),
    ],
)
def test_cloud_base_derives_final_output_and_local_cache(
    tmp_path: Path, base: str, expected: str
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        f"paths: {{work_dir: '{base}', cache_dir: './cache'}}\nos_downloads: {{package_id: '1', version_id: '1'}}",
    )
    settings = load_settings(config, load_env=False)
    assert settings.paths.output_dir == expected
    assert settings.paths.work_dir == (tmp_path / "cache").resolve()
    assert settings.paths.downloads_dir == tmp_path / "cache/downloads"
    assert settings.paths.extracted_dir == tmp_path / "cache/extracted"
    assert settings.paths.parquet_dir == tmp_path / "cache/parquet"


def test_cloud_output_directory_needs_no_uri_option(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        "paths: {work_dir: './data', overrides: {output_dir: 's3://test-bucket/final/'}}\nos_downloads: {package_id: '1', version_id: '1'}",
    )
    settings = load_settings(config, load_env=False)
    assert settings.paths.output_dir == "s3://test-bucket/final"
    assert settings.paths.work_dir == tmp_path / "data"


def test_remote_intermediates_are_rejected_explicitly(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        "paths: {overrides: {extracted_dir: 's3://test-bucket/raw'}}\nos_downloads: {package_id: '1', version_id: '1'}",
    )
    with pytest.raises(SettingsError, match="must be local"):
        load_settings(config, load_env=False)


@pytest.mark.parametrize(
    ("uri", "provider"),
    [
        ("s3://test-bucket/output/", "s3"),
        ("az://testaccount.blob.core.windows.net/outputs/ngd/", "azure"),
    ],
)
def test_cloud_output_normalizes_uri(uri: str, provider: str) -> None:
    assert resolve_output_path(Path.cwd(), uri) == uri.rstrip("/")


@pytest.mark.parametrize(
    "uri",
    [
        "s3://test-bucket",
        "s3://test-bucket/",
        "s3://test-bucket/../output",
        "s3://test-bucket/output/*",
        "s3://test-bucket/output?token=secret",
        "s3://user@test-bucket/output",
        "s3://test-bucket:443/output",
        "s3://test-bucket/output/%2e%2e",
        "s3://test-bucket/output//ngd",
        "az://testaccount.blob.core.windows.net/outputs",
        "az://outputs/ngd",
    ],
)
def test_cloud_output_rejects_unsafe_uri(uri: str) -> None:
    with pytest.raises(SettingsError):
        resolve_output_path(Path.cwd(), uri)


def test_cloud_output_rejects_aws_options_on_azure(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(
        config,
        "paths: {work_dir: 'az://testaccount.blob.core.windows.net/outputs'}\nos_downloads: {package_id: '1', version_id: '1'}\noutput: {aws: {region: 'eu-west-2'}}",
    )
    with pytest.raises(SettingsError) as error:
        load_settings(config, load_env=False)
    assert "output.aws" in str(error.value.validation_error)


def test_cloud_output_loads_without_resolving_uri_as_local_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
    paths:
      work_dir: ./data
      overrides: {output_dir: 's3://test-bucket/output'}
    os_downloads:
      package_id: "1"
      version_id: "1"
    output:
      aws: {region: eu-west-2, profile: development, credential_chain: [sso]}
  """,
    )
    settings = load_settings(config_path, load_env=False)
    assert settings.paths.output_dir == "s3://test-bucket/output"
    assert settings.output.aws.profile == "development"
    assert settings.paths.work_dir == (tmp_path / "data").resolve()


def test_load_settings_resolves_paths_relative_to_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data

        os_downloads:
          package_id: "16465"
          version_id: "104444"

        processing:
          num_chunks: 2
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.paths.work_dir == (tmp_path / "data").resolve()
    assert settings.paths.downloads_dir == (tmp_path / "data/downloads").resolve()
    assert settings.processing.num_chunks == 2


def test_load_settings_rejects_unknown_config_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data

        os_downloads:
          package_id: "16465"
          version_id: "104444"
          extra_key: nope
        """,
    )

    with pytest.raises(SettingsError) as exc_info:
        load_settings(config_path, load_env=False)

    assert exc_info.value.validation_error is not None
    assert isinstance(exc_info.value.validation_error, ValidationError)


def test_load_settings_missing_package_id_has_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data

        os_downloads:
          version_id: "104444"
        """,
    )

    with pytest.raises(SettingsError) as exc_info:
        load_settings(config_path, load_env=False)

    assert str(exc_info.value) == "Invalid configuration"
    assert exc_info.value.validation_error is not None


def test_load_settings_uses_work_dir_for_default_subpaths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./custom_data

        os_downloads:
          package_id: "16465"
          version_id: "104444"
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.paths.work_dir == (tmp_path / "custom_data").resolve()
    assert settings.paths.downloads_dir == (tmp_path / "custom_data/downloads").resolve()
    assert settings.paths.extracted_dir == (tmp_path / "custom_data/extracted").resolve()
    assert settings.paths.parquet_dir == (tmp_path / "custom_data/parquet").resolve()
    assert settings.paths.output_dir == (tmp_path / "custom_data/output").resolve()


def test_load_settings_allows_missing_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OS_PROJECT_API_KEY", raising=False)
    monkeypatch.delenv("OS_PROJECT_API_SECRET", raising=False)

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data

        os_downloads:
          package_id: "16465"
          version_id: "104444"
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.os_downloads.api_key is None
    assert settings.os_downloads.api_secret is None


def test_load_settings_validates_positive_read_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data

        os_downloads:
          package_id: "16465"
          version_id: "104444"
          read_timeout_seconds: 0
        """,
    )

    with pytest.raises(SettingsError) as exc_info:
        load_settings(config_path, load_env=False)

    assert exc_info.value.validation_error is not None
    assert "read_timeout_seconds" in str(exc_info.value.validation_error)


def test_load_settings_defaults_source_and_num_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        os_downloads:
          package_id: "16465"
          version_id: "104444"
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.source.type == "ngd"
    assert settings.processing.num_chunks == 10
    assert settings.processing.sort_output_by_postcode is True
    assert settings.processing.ngd_excluded_stems == ["historicaddress"]
    assert settings.processing.abp_excluded_logical_statuses == [8]


def test_load_settings_validates_ngd_excluded_stems(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        os_downloads:
          package_id: "16465"
          version_id: "104444"

        processing:
          ngd_excluded_stems:
            - HistoricAddress
            - prebuildaddress
            - historicaddress
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.processing.ngd_excluded_stems == ["historicaddress", "prebuildaddress"]


def test_load_settings_rejects_invalid_ngd_excluded_stem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        os_downloads:
          package_id: "16465"
          version_id: "104444"

        processing:
          ngd_excluded_stems:
            - not-a-feature
        """,
    )

    with pytest.raises(SettingsError) as exc_info:
        load_settings(config_path, load_env=False)

    assert exc_info.value.validation_error is not None
    assert "ngd_excluded_stems" in str(exc_info.value.validation_error)


def test_load_settings_validates_abp_excluded_logical_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        os_downloads:
          package_id: "16465"
          version_id: "104444"

        processing:
          abp_excluded_logical_statuses:
            - "8"
            - 3
            - 8
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.processing.abp_excluded_logical_statuses == [8, 3]


def test_load_settings_rejects_invalid_abp_excluded_logical_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        os_downloads:
          package_id: "16465"
          version_id: "104444"

        processing:
          abp_excluded_logical_statuses:
            - 2
        """,
    )

    with pytest.raises(SettingsError) as exc_info:
        load_settings(config_path, load_env=False)

    assert exc_info.value.validation_error is not None
    assert "abp_excluded_logical_statuses" in str(exc_info.value.validation_error)


def test_load_settings_applies_path_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data
          overrides:
            downloads_dir: ./custom/downloads
            extracted_dir: /tmp/extracted

        os_downloads:
          package_id: "16465"
          version_id: "104444"
        """,
    )

    settings = load_settings(config_path, load_env=False)

    assert settings.paths.work_dir == (tmp_path / "data").resolve()
    assert settings.paths.downloads_dir == (tmp_path / "custom/downloads").resolve()
    assert str(settings.paths.extracted_dir).endswith("/tmp/extracted")
    assert settings.paths.parquet_dir == (tmp_path / "data/parquet").resolve()
    assert settings.paths.output_dir == (tmp_path / "data/output").resolve()


def test_load_settings_rejects_legacy_path_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OS_PROJECT_API_KEY", "key")
    monkeypatch.setenv("OS_PROJECT_API_SECRET", "secret")

    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        paths:
          work_dir: ./data
          downloads_dir: ./legacy/downloads

        os_downloads:
          package_id: "16465"
          version_id: "104444"
        """,
    )

    with pytest.raises(SettingsError, match="no longer supported"):
        load_settings(config_path, load_env=False)
