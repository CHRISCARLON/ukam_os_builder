"""Final output destinations and fixed-prefix cloud publication."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import duckdb
import fsspec
from fsspec.spec import AbstractFileSystem
from pydantic import Field, ValidationError

from ukam_os_builder._exceptions import ToFlatfileError
from ukam_os_builder.api.settings import (
    Settings,
    SettingsError,
    StrictBaseModel,
    create_duckdb_connection,
)
from ukam_os_builder.os_builder.extract import _sql_string

logger = logging.getLogger(__name__)


class _ChunkMetadata(StrictBaseModel):
    name: str
    rows: int = Field(ge=0, strict=True)
    size_bytes: int = Field(gt=0, strict=True)


class _OutputManifest(StrictBaseModel):
    schema_version: Literal[1] = 1
    source: Literal["ngd", "abp"]
    package_id: str
    version_id: str
    signature: str
    num_chunks: int = Field(ge=1, strict=True)
    chunks: list[_ChunkMetadata]


class Writer(ABC):
    """Strategy for native Parquet writes and destination lifecycle."""

    is_remote = False

    def __init__(self, settings: Settings, source: Literal["ngd", "abp"]) -> None:
        self.settings = settings
        self.source = source
        self.output = settings.output
        self._filesystem: AbstractFileSystem | None = None
        self._chunks: list[_ChunkMetadata] = []
        self._owned_name = re.compile(
            rf"{source}_for_uk_address_matcher\.chunk_\d{{3,}}_of_\d{{3,}}\.parquet"
        )
        self.manifest_name = f"_ukam_{source}_output.json"
        self.location = str(settings.paths.output_dir)
        self.root = self.location

    @abstractmethod
    def target(self, name: str) -> Path | str:
        raise NotImplementedError

    @property
    @abstractmethod
    def filesystem(self) -> AbstractFileSystem:
        raise NotImplementedError

    @abstractmethod
    def configure_connection(self, connection: duckdb.DuckDBPyConnection) -> None:
        raise NotImplementedError

    @abstractmethod
    def existing_outputs(self, force: bool) -> list[Path | str] | None:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, connection: duckdb.DuckDBPyConnection, force: bool) -> None:
        raise NotImplementedError

    def completed_manifest(self) -> _OutputManifest | None:
        return None

    def record_chunk(self, chunk_index: int, rows: int) -> None:
        return None

    def finish(self) -> None:
        return None

    def chunk_name(self, chunk_index: int) -> str:
        return (
            f"{self.source}_for_uk_address_matcher.chunk_{chunk_index + 1:03d}"
            f"_of_{self.settings.processing.num_chunks:03d}.parquet"
        )

    @contextmanager
    def connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        self.settings.paths.work_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="ukam_duckdb_", dir=self.settings.paths.work_dir) as scratch:
            connection = create_duckdb_connection(self.settings)
            try:
                connection.execute(
                    f"SET temp_directory = '{_sql_string(Path(scratch).as_posix())}'"
                )
                yield connection
            finally:
                connection.close()

    def write_query(
        self,
        connection: duckdb.DuckDBPyConnection,
        query: str,
        chunk_index: int,
        rows: int,
        *,
        compression: str,
        compression_level: int,
    ) -> Path | str:
        target = self.target(self.chunk_name(chunk_index))
        if isinstance(target, Path) and target.exists():
            target.unlink()
        connection.execute(
            f"COPY ({query}) TO '{_sql_string(str(target))}' (FORMAT PARQUET, "
            f"COMPRESSION '{_sql_string(compression)}', COMPRESSION_LEVEL {compression_level})"
        )
        self.record_chunk(chunk_index, rows)
        return target

    def write_relation(
        self, relation: duckdb.DuckDBPyRelation, chunk_index: int, rows: int
    ) -> Path | str:
        target = self.target(self.chunk_name(chunk_index))
        if isinstance(target, Path) and target.exists():
            target.unlink()
        relation.write_parquet(str(target))
        self.record_chunk(chunk_index, rows)
        return target


class _CloudWriter(Writer):
    """Fixed-prefix publication shared by cloud filesystem strategies."""

    is_remote = True
    protocol: str
    extra: str
    extensions: tuple[str, ...]

    def target(self, name: str) -> str:
        return f"{self.location}/{name}"

    @abstractmethod
    def _filesystem_options(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _secret_options(self, connection: duckdb.DuckDBPyConnection) -> list[str]:
        raise NotImplementedError

    @property
    def filesystem(self) -> AbstractFileSystem:
        if self._filesystem is None:
            try:
                self._filesystem = fsspec.filesystem(
                    self.protocol,
                    anon=False,
                    skip_instance_cache=True,
                    **self._filesystem_options(),
                )
            except ImportError as exc:
                raise SettingsError(
                    f'Cloud output requires: pip install "ukam-os-builder[{self.extra}]"'
                ) from exc
        return self._filesystem

    def configure_connection(self, connection: duckdb.DuckDBPyConnection) -> None:
        found = re.match(r"(\d+)\.(\d+)\.(\d+)", duckdb.__version__)
        if found is None or tuple(map(int, found.groups())) < (1, 4, 3):
            raise SettingsError("Cloud output requires duckdb>=1.4.3; install the provider extra")
        for extension in self.extensions:
            connection.execute(f"INSTALL {extension}")
            connection.execute(f"LOAD {extension}")
        options = [
            "PROVIDER credential_chain",
            f"SCOPE '{_sql_string(self.location + '/')}'",
            *self._secret_options(connection),
        ]
        connection.execute(f"CREATE OR REPLACE TEMPORARY SECRET ukam_output ({', '.join(options)})")

    def _owned_files(self) -> dict[str, Path | str]:
        pattern = f"{self.source}_for_uk_address_matcher.chunk_*.parquet"
        filesystem = self.filesystem
        filesystem.invalidate_cache(self.root)
        result: dict[str, Path | str] = {}
        for path in filesystem.glob(f"{self.root}/{pattern}"):
            normalized = filesystem._strip_protocol(path).lstrip("/")
            parent, _, name = normalized.rpartition("/")
            if parent == self.root and self._owned_name.fullmatch(name):
                result[name] = path
        return result

    def _signature(self) -> str:
        processing = self.settings.processing.model_dump()
        processing.pop("duckdb_memory_limit", None)
        payload = {
            "source": self.source,
            "package_id": self.settings.os_downloads.package_id,
            "version_id": self.settings.os_downloads.version_id,
            "processing": processing,
            "builder_version": version("ukam-os-builder"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def completed_manifest(self) -> _OutputManifest | None:
        try:
            data = self.filesystem.cat_file(f"{self.root}/{self.manifest_name}")
        except FileNotFoundError:
            return None
        try:
            manifest = _OutputManifest.model_validate_json(data)
        except (ValidationError, ValueError) as exc:
            raise ToFlatfileError(
                "Invalid cloud completion manifest; run with --overwrite"
            ) from exc
        names = [self.chunk_name(index) for index in range(self.settings.processing.num_chunks)]
        files = self._owned_files()
        if (
            manifest.source != self.source
            or manifest.package_id != self.settings.os_downloads.package_id
            or manifest.version_id != self.settings.os_downloads.version_id
            or manifest.signature != self._signature()
            or manifest.num_chunks != self.settings.processing.num_chunks
            or [chunk.name for chunk in manifest.chunks] != names
            or set(files) != set(names)
            or any(
                self.filesystem.size(str(files[chunk.name])) != chunk.size_bytes
                for chunk in manifest.chunks
            )
        ):
            raise ToFlatfileError(
                "Cloud output is incomplete or differs from this configuration; run with --overwrite"
            )
        return manifest

    def existing_outputs(self, force: bool) -> list[Path | str] | None:
        if force:
            return None
        manifest = self.completed_manifest()
        if manifest:
            return [self.target(chunk.name) for chunk in manifest.chunks]
        if self._owned_files():
            raise ToFlatfileError(
                "Cloud chunks exist without a completed run; run with --overwrite"
            )
        return None

    def prepare(self, connection: duckdb.DuckDBPyConnection, force: bool) -> None:
        self._chunks.clear()
        if not force and self.filesystem.exists(f"{self.root}/{self.manifest_name}"):
            raise ToFlatfileError("Cloud completion manifest already exists; run with --overwrite")
        self.configure_connection(connection)
        filesystem = self.filesystem
        probe_name = f"_ukam_probe_{uuid4().hex}.parquet"
        probe_key = f"{self.root}/{probe_name}"
        try:
            connection.execute(
                f"COPY (SELECT 1 AS permission_probe) TO '{_sql_string(str(self.target(probe_name)))}' (FORMAT PARQUET)"
            )
            filesystem.rm(probe_key)
        except Exception:
            if filesystem.exists(probe_key):
                filesystem.rm(probe_key)
            raise
        filesystem.invalidate_cache(self.root)
        if filesystem.exists(probe_key):
            raise ToFlatfileError(
                "Cloud permission probe could not be deleted; refusing to overwrite"
            )
        files = self._owned_files()
        if not force and files:
            raise ToFlatfileError("Cloud chunks already exist; run with --overwrite")
        if force:
            manifest_key = f"{self.root}/{self.manifest_name}"
            with suppress(FileNotFoundError):
                filesystem.rm(manifest_key)
            filesystem.invalidate_cache(self.root)
            if filesystem.exists(manifest_key):
                raise ToFlatfileError("Cloud completion manifest could not be deleted")
            for path in files.values():
                filesystem.rm(str(path))
            if self._owned_files():
                raise ToFlatfileError("Cloud chunks could not be deleted; refusing to overwrite")

    def record_chunk(self, chunk_index: int, rows: int) -> None:
        name = self.chunk_name(chunk_index)
        self._chunks.append(
            _ChunkMetadata(
                name=name,
                rows=rows,
                size_bytes=self.filesystem.size(f"{self.root}/{name}"),
            )
        )

    def finish(self) -> None:
        expected = [self.chunk_name(index) for index in range(self.settings.processing.num_chunks)]
        if [chunk.name for chunk in self._chunks] != expected or set(self._owned_files()) != set(
            expected
        ):
            raise ToFlatfileError(
                "Not all cloud chunks were written; completion manifest was not published"
            )
        manifest = _OutputManifest(
            source=self.source,
            package_id=self.settings.os_downloads.package_id,
            version_id=self.settings.os_downloads.version_id,
            signature=self._signature(),
            num_chunks=self.settings.processing.num_chunks,
            chunks=self._chunks,
        )
        key = f"{self.root}/{self.manifest_name}"
        try:
            self.filesystem.pipe_file(key, manifest.model_dump_json().encode())
        except Exception:
            with suppress(FileNotFoundError):
                self.filesystem.rm(key)
            raise
        self.filesystem.invalidate_cache(self.root)


class LocalWriter(Writer):
    def __init__(self, settings: Settings, source: Literal["ngd", "abp"]) -> None:
        super().__init__(settings, source)
        directory = Path(settings.paths.output_dir)
        self.directory = (
            directory if directory.is_absolute() else settings.config_path.parent / directory
        ).resolve()
        self.location = self.directory.as_posix()
        self.root = self.location

    def target(self, name: str) -> Path:
        return self.directory / name

    @property
    def filesystem(self) -> AbstractFileSystem:
        raise ValueError("Local destinations do not need a cloud filesystem")

    def configure_connection(self, connection: duckdb.DuckDBPyConnection) -> None:
        pass

    def _owned_files(self) -> dict[str, Path]:
        return {
            path.name: path
            for path in self.directory.glob(f"{self.source}_for_uk_address_matcher.chunk_*.parquet")
            if path.is_file() and self._owned_name.fullmatch(path.name)
        }

    def existing_outputs(self, force: bool) -> list[Path | str] | None:
        if force:
            return None
        files = self._owned_files()
        return [files[name] for name in sorted(files)] or None

    def prepare(self, connection: duckdb.DuckDBPyConnection, force: bool) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if force:
            for path in self._owned_files().values():
                path.unlink()


class S3Writer(_CloudWriter):
    protocol = "s3"
    extra = "aws"
    extensions = ("httpfs", "aws")

    def __init__(self, settings: Settings, source: Literal["ngd", "abp"]) -> None:
        super().__init__(settings, source)
        parsed = urlsplit(self.location)
        self.root = f"{parsed.netloc}{parsed.path}"

    def _filesystem_options(self) -> dict:
        options: dict = {}
        aws = self.output.aws if self.output else None
        region = (
            (aws.region if aws else None)
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        if region:
            options["client_kwargs"] = {"region_name": region}
        if aws:
            if aws.profile:
                options["profile"] = aws.profile
            if aws.credential_chain:
                from aiobotocore.credentials import AioEnvProvider
                from aiobotocore.session import AioSession

                session = AioSession(profile=aws.profile)
                resolver = session.get_component("credential_provider")
                providers = {provider.METHOD: provider for provider in resolver.providers}
                providers.setdefault("env", AioEnvProvider())
                methods = {
                    "env": ["env"],
                    "config": ["assume-role", "shared-credentials-file", "config-file"],
                    "sso": ["sso"],
                    "instance": ["iam-role"],
                    "process": ["custom-process"],
                }
                resolver.providers = [
                    providers[method]
                    for chain in aws.credential_chain
                    for method in methods[chain]
                    if method in providers
                ]
                options["session"] = session
            if aws.kms_key_id:
                options["s3_additional_kwargs"] = {
                    "ServerSideEncryption": "aws:kms",
                    "SSEKMSKeyId": aws.kms_key_id,
                }
        return options

    def _secret_options(self, connection: duckdb.DuckDBPyConnection) -> list[str]:
        options = ["TYPE s3", "REFRESH auto"]
        aws = self.output.aws if self.output else None
        if aws:
            if aws.credential_chain:
                options.append(f"CHAIN '{';'.join(aws.credential_chain)}'")
            for option, value in [("REGION", aws.region), ("PROFILE", aws.profile)]:
                if value:
                    options.append(f"{option} '{_sql_string(value)}'")
            if aws.kms_key_id:
                connection.execute(f"SET s3_kms_key_id = '{_sql_string(aws.kms_key_id)}'")
        return options


class AzureWriter(_CloudWriter):
    protocol = "az"
    extra = "azure"
    extensions = ("azure",)

    def __init__(self, settings: Settings, source: Literal["ngd", "abp"]) -> None:
        super().__init__(settings, source)
        parsed = urlsplit(self.location)
        self.account_name = parsed.netloc.split(".", 1)[0]
        self.root = parsed.path.lstrip("/")

    def _filesystem_options(self) -> dict:
        return {"account_name": self.account_name}

    def _secret_options(self, connection: duckdb.DuckDBPyConnection) -> list[str]:
        return ["TYPE azure", f"ACCOUNT_NAME '{_sql_string(self.account_name)}'"]


WRITERS: dict[str, type[Writer]] = {"s3": S3Writer, "az": AzureWriter}


def build_writer(settings: Settings, source: Literal["ngd", "abp"]) -> Writer:
    """Select a writer by path scheme, falling back to logged local filesystem output."""
    scheme = urlsplit(str(settings.paths.output_dir)).scheme
    writer = WRITERS.get(scheme, LocalWriter)(settings, source)
    if not writer.is_remote:
        logger.info("Writing output locally: %s", writer.location)
    return writer
