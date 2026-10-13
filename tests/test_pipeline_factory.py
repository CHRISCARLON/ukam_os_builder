from __future__ import annotations

import logging

import pytest

from ukam_os_builder._exceptions import PipelineError
from ukam_os_builder.api.settings import Settings
from ukam_os_builder.os_builder.pipeline_factory import (
    PipelineDefinition,
    make_standard_step,
    run_pipeline,
)


def test_run_pipeline_executes_declared_individual_step(temp_settings: Settings) -> None:
    calls: list[tuple[object, bool]] = []

    def run_split(settings: object, force: bool) -> None:
        calls.append((settings, force))

    definition = PipelineDefinition(
        dataset_name="abp",
        steps=(make_standard_step("split", run_split),),
        clean_patterns={},
        step_outputs={},
    )
    run_pipeline(
        definition=definition,
        step="split",
        settings=temp_settings,
        logger=logging.getLogger("test.pipeline_factory"),
    )

    assert calls == [(temp_settings, False)]


def test_run_pipeline_rejects_step_not_declared_by_pipeline(temp_settings: Settings) -> None:
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
            settings=temp_settings,
            logger=logging.getLogger("test.pipeline_factory"),
        )


def test_run_pipeline_rejects_list_only_for_non_download_step(temp_settings: Settings) -> None:
    definition = PipelineDefinition(
        dataset_name="abp",
        steps=(make_standard_step("split", lambda _settings, _force: None),),
        clean_patterns={},
        step_outputs={},
    )

    with pytest.raises(PipelineError, match="list_only can only be used"):
        run_pipeline(
            definition=definition,
            step="split",
            settings=temp_settings,
            list_only=True,
            logger=logging.getLogger("test.pipeline_factory"),
        )
