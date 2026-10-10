from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from ukam_os_builder._exceptions import PipelineError
from ukam_os_builder.os_builder.pipeline_factory import (
    PipelineDefinition,
    make_standard_step,
    run_pipeline,
)


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        paths=SimpleNamespace(work_dir=tmp_path),
    )


def test_run_pipeline_executes_declared_individual_step(tmp_path: Path) -> None:
    calls: list[tuple[object, bool]] = []

    def run_split(settings: object, force: bool) -> None:
        calls.append((settings, force))

    definition = PipelineDefinition(
        dataset_name="abp",
        steps=(make_standard_step("split", run_split),),
        clean_patterns={},
        step_outputs={},
    )
    settings = _settings(tmp_path)

    run_pipeline(
        definition=definition,
        step="split",
        settings=settings,
        logger=logging.getLogger("test.pipeline_factory"),
    )

    assert calls == [(settings, False)]


def test_run_pipeline_rejects_step_not_declared_by_pipeline(tmp_path: Path) -> None:
    definition = PipelineDefinition(
        dataset_name="ngd",
        steps=(make_standard_step("flatfile", lambda _settings, _force: None),),
        clean_patterns={},
        step_outputs={},
    )

    with pytest.raises(PipelineError, match="Invalid step: split"):
        run_pipeline(
            definition=definition,
            step="split",
            settings=_settings(tmp_path),
            logger=logging.getLogger("test.pipeline_factory"),
        )
