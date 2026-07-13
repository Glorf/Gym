# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import MagicMock

import pytest
from fastapi import Body
from fastapi.testclient import TestClient
from pydantic import ConfigDict, ValidationError

from nemo_gym.agent_execution_capture import (
    AgentExecutionCapture,
    AgentExecutionCaptureStore,
    AgentExecutionCoverage,
    AgentExecutionRecorder,
    AgentInvocationRecord,
    ModelCallLink,
    ToolSpanRecord,
    clear_agent_execution_captures_for_rollouts,
    merge_agent_execution_capture_into_record,
)
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.server_utils import ServerClient


EXACT_COVERAGE = AgentExecutionCoverage(
    lineage="exact",
    model_call_attribution="exact",
    tool_timing="exact",
)


def test_recorder_captures_individual_span_with_injected_clock() -> None:
    ticks = iter([10, 40])
    recorder = AgentExecutionRecorder("2-3", "simple_agent", EXACT_COVERAGE, clock=lambda: next(ticks))

    with recorder.tool_span(
        tool_call_ids=["call-1"],
        granularity="individual",
        measurement_scope="caller_round_trip",
        model_server="policy",
        model_response_id="response-1",
    ):
        pass

    capture = recorder.capture()
    assert capture.rollout_id == "2-3"
    assert capture.agent_invocations == [AgentInvocationRecord(id="root", source="simple_agent")]
    assert capture.tool_spans[0].model_dump() == {
        "id": "tool-0",
        "agent_invocation_id": "root",
        "model_server": "policy",
        "model_response_id": "response-1",
        "tool_call_ids": ["call-1"],
        "granularity": "individual",
        "measurement_scope": "caller_round_trip",
        "clock_id": recorder.clock_id,
        "started_ns": 10,
        "ended_ns": 40,
        "status": "returned",
    }


def test_tool_span_records_error_without_swallowing_it() -> None:
    ticks = iter([100, 160])
    recorder = AgentExecutionRecorder("0-0", "agent", EXACT_COVERAGE, clock=lambda: next(ticks))

    with pytest.raises(RuntimeError, match="tool failed"):
        with recorder.tool_span(
            tool_call_ids=["call"],
            granularity="individual",
            measurement_scope="caller_round_trip",
        ):
            raise RuntimeError("tool failed")

    span = recorder.capture().tool_spans[0]
    assert (span.started_ns, span.ended_ns, span.status) == (100, 160, "raised")


def test_span_recording_failure_preserves_original_tool_error() -> None:
    clock_reads = 0

    def failing_clock() -> int:
        nonlocal clock_reads
        clock_reads += 1
        if clock_reads == 1:
            return 100
        raise RuntimeError("clock failed")

    recorder = AgentExecutionRecorder("0-0", "agent", EXACT_COVERAGE, clock=failing_clock)
    with pytest.raises(ValueError, match="tool failed"):
        with recorder.tool_span(
            tool_call_ids=["call"],
            granularity="individual",
            measurement_scope="caller_round_trip",
        ):
            raise ValueError("tool failed")

    capture = recorder.capture()
    assert capture.tool_spans == []
    assert capture.capture_complete is False
    assert capture.warnings == ["tool_span_recording_failed"]


@pytest.mark.asyncio
async def test_recorder_preserves_overlapping_tool_intervals() -> None:
    ticks = iter([10, 20, 50, 60])
    recorder = AgentExecutionRecorder("0-0", "agent", EXACT_COVERAGE, clock=lambda: next(ticks))
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def execute(tool_call_id: str) -> None:
        nonlocal started
        with recorder.tool_span(
            tool_call_ids=[tool_call_id],
            granularity="individual",
            measurement_scope="caller_round_trip",
        ):
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()

    tasks = [asyncio.create_task(execute("a")), asyncio.create_task(execute("b"))]
    await both_started.wait()
    release.set()
    await asyncio.gather(*tasks)

    spans = recorder.capture().tool_spans
    assert {span.tool_call_ids[0] for span in spans} == {"a", "b"}
    assert max(span.started_ns for span in spans) < min(span.ended_ns for span in spans)


def test_individual_span_rejects_multiple_tool_calls() -> None:
    with pytest.raises(ValidationError, match="exactly one tool call"):
        ToolSpanRecord(
            id="span",
            agent_invocation_id="root",
            tool_call_ids=["a", "b"],
            granularity="individual",
            measurement_scope="caller_round_trip",
            clock_id="clock",
            started_ns=1,
            ended_ns=2,
            status="returned",
        )


def test_capture_validates_tree_and_record_owners() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        AgentExecutionCapture(
            rollout_id="0-0",
            agent_server="agent",
            coverage=EXACT_COVERAGE,
            agent_invocations=[
                AgentInvocationRecord(id="root", source="agent"),
                AgentInvocationRecord(id="a", parent_id="b", source="agent"),
                AgentInvocationRecord(id="b", parent_id="a", source="agent"),
            ],
        )

    with pytest.raises(ValidationError, match="unknown agent invocation"):
        AgentExecutionCapture(
            rollout_id="0-0",
            agent_server="agent",
            coverage=EXACT_COVERAGE,
            agent_invocations=[AgentInvocationRecord(id="root", source="agent")],
            model_call_links=[ModelCallLink(agent_invocation_id="missing", response_id="r")],
        )


def test_capture_rejects_multiple_roots() -> None:
    with pytest.raises(ValidationError, match="exactly one root"):
        AgentExecutionCapture(
            rollout_id="0-0",
            agent_server="agent",
            coverage=EXACT_COVERAGE,
            agent_invocations=[
                AgentInvocationRecord(id="root-a", source="agent"),
                AgentInvocationRecord(id="root-b", source="agent"),
            ],
        )


def test_recorder_emits_explicit_agent_parent_edges() -> None:
    recorder = AgentExecutionRecorder("0-0", "agent", EXACT_COVERAGE)
    recorder.add_agent_invocation("child", source="subagent")
    recorder.add_agent_invocation("grandchild", source="subagent", parent_id="child")

    assert [invocation.model_dump() for invocation in recorder.capture().agent_invocations] == [
        {
            "id": "root",
            "parent_id": None,
            "source": "agent",
        },
        {
            "id": "child",
            "parent_id": "root",
            "source": "subagent",
        },
        {
            "id": "grandchild",
            "parent_id": "child",
            "source": "subagent",
        },
    ]


def test_recorder_adopts_sole_worker_capture() -> None:
    owner = AgentExecutionRecorder("0-0", "agent", EXACT_COVERAGE)
    worker = AgentExecutionRecorder(
        "0-0",
        "agent",
        AgentExecutionCoverage(lineage="partial", model_call_attribution="partial", tool_timing="exact"),
        clock=iter([10, 20]).__next__,
    )
    worker.add_agent_invocation("child", source="subagent")
    with worker.tool_span(
        tool_call_ids=["call-1"],
        granularity="individual",
        measurement_scope="caller_round_trip",
        agent_invocation_id="child",
    ):
        pass
    worker.mark_incomplete("worker_partial")

    worker_capture = worker.capture()
    owner.adopt(worker_capture)

    assert owner.capture().coverage == worker_capture.coverage
    assert owner.capture().capture_complete is False
    assert owner.capture().warnings == ["worker_partial"]
    assert owner.capture().tool_spans == worker_capture.tool_spans
    assert owner.capture().agent_invocations == worker_capture.agent_invocations

    other_rollout = worker_capture.model_copy(update={"rollout_id": "1-0"})
    with pytest.raises(ValueError, match="rollout_id"):
        owner.adopt(other_rollout)

    other_server = worker_capture.model_copy(update={"agent_server": "other"})
    with pytest.raises(ValueError, match="agent_server"):
        owner.adopt(other_server)


def test_store_round_trip_and_clear(tmp_path: Path) -> None:
    recorder = AgentExecutionRecorder("4-1", "agent", EXACT_COVERAGE)
    store = AgentExecutionCaptureStore(tmp_path)
    store.write(recorder.capture())

    assert store.read("4-1") == recorder.capture()
    assert list(tmp_path.glob(".*")) == []

    store.clear("4-1")
    assert store.read("4-1") is None


def test_merge_attaches_raw_capture_without_rewriting_links(tmp_path: Path) -> None:
    recorder = AgentExecutionRecorder("1-2", "agent", EXACT_COVERAGE)
    recorder.add_model_call_link(response_id="r0", model_server="planner")
    recorder.record_tool_span(
        tool_call_ids=["tool-17"],
        granularity="individual",
        measurement_scope="caller_round_trip",
        model_server="planner",
        model_response_id="r0",
        started_ns=1,
        ended_ns=2,
        status="returned",
    )
    AgentExecutionCaptureStore(tmp_path).write(recorder.capture())
    record = {
        "_ng_task_index": 1,
        "_ng_rollout_index": 2,
        "ng_model_call_capture": {"calls": [{"call_index": 0, "model_server": "planner", "response_id": "r0"}]},
    }

    merge_agent_execution_capture_into_record(record, [tmp_path])

    capture = record["ng_agent_execution_capture"]
    assert capture["model_call_links"] == [
        {
            "agent_invocation_id": "root",
            "model_server": "planner",
            "response_id": "r0",
        }
    ]
    span = capture["tool_spans"][0]
    assert span["model_server"] == "planner"
    assert span["model_response_id"] == "r0"
    assert span["tool_call_ids"] == ["tool-17"]


def test_corrupt_capture_does_not_change_rollout_record(tmp_path: Path) -> None:
    AgentExecutionCaptureStore(tmp_path).path_for("1-2").write_text("not-json")
    record = {"_ng_task_index": 1, "_ng_rollout_index": 2, "reward": 1.0}

    assert merge_agent_execution_capture_into_record(record, [tmp_path]) == record
    assert "ng_agent_execution_capture" not in record


def test_clear_agent_capture_uses_rollout_indices(tmp_path: Path) -> None:
    store = AgentExecutionCaptureStore(tmp_path)
    store.write(AgentExecutionRecorder("7-8", "agent", EXACT_COVERAGE).capture())

    clear_agent_execution_captures_for_rollouts(
        [{"_ng_task_index": 7, "_ng_rollout_index": 8}],
        [tmp_path],
    )

    assert store.read("7-8") is None


class _RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


def _capture_agent(
    tmp_path: Path,
    *,
    enabled: bool = True,
    run_raises: bool = False,
    num_workers: int | None = None,
    requires_single_worker: bool = False,
) -> SimpleResponsesAPIAgent:
    capture_expected = enabled and not (requires_single_worker and (num_workers or 1) > 1)

    class _Agent(SimpleResponsesAPIAgent):
        def agent_execution_coverage(self):
            return EXACT_COVERAGE

        def agent_execution_capture_requires_single_worker(self) -> bool:
            return requires_single_worker

        async def responses(self, body=Body()):
            raise NotImplementedError

        async def run(self, body: _RunRequest = Body()):
            recorder = self.agent_execution_recorder_for_run(body)
            assert (recorder is not None) is capture_expected
            if run_raises:
                raise RuntimeError("run failed")
            return {"ok": True}

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {
        "observability_enabled": enabled,
        "model_call_capture_dir": str(tmp_path),
    }
    config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent", num_workers=num_workers)
    return _Agent(config=config, server_client=server_client)


def test_run_route_owns_and_flushes_one_capture(tmp_path: Path) -> None:
    agent = _capture_agent(tmp_path)

    response = TestClient(agent.setup_webserver()).post(
        "/run",
        json={
            "responses_create_params": {"input": []},
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
        },
    )

    assert response.status_code == 200
    assert AgentExecutionCaptureStore(tmp_path).read("3-4") is not None


def test_run_route_capture_disabled_has_no_store_or_clock_work(tmp_path: Path) -> None:
    agent = _capture_agent(tmp_path, enabled=False)

    response = TestClient(agent.setup_webserver()).post(
        "/run",
        json={
            "responses_create_params": {"input": []},
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
        },
    )

    assert response.status_code == 200
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_process_affine_capture_is_disabled_with_multiple_workers(tmp_path: Path) -> None:
    agent = _capture_agent(tmp_path, num_workers=2, requires_single_worker=True)

    response = TestClient(agent.setup_webserver()).post(
        "/run",
        json={
            "responses_create_params": {"input": []},
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
        },
    )

    assert response.status_code == 200
    capture = AgentExecutionCaptureStore(tmp_path).read("3-4")
    assert capture is not None
    assert capture.coverage == AgentExecutionCoverage(
        lineage="unavailable",
        model_call_attribution="unavailable",
        tool_timing="unavailable",
    )
    assert capture.capture_complete is False
    assert capture.warnings == ["agent_execution_capture_requires_single_worker"]


def test_run_failure_flushes_capture_without_changing_exception(tmp_path: Path) -> None:
    agent = _capture_agent(tmp_path, run_raises=True)

    with pytest.raises(RuntimeError, match="run failed"):
        TestClient(agent.setup_webserver()).post(
            "/run",
            json={
                "responses_create_params": {"input": []},
                "_ng_task_index": 3,
                "_ng_rollout_index": 4,
            },
        )

    capture = AgentExecutionCaptureStore(tmp_path).read("3-4")
    assert capture is not None
    assert capture.capture_complete is True
    assert capture.warnings == []


def test_concurrent_duplicate_owner_marks_persisted_capture_incomplete(tmp_path: Path) -> None:
    agent = _capture_agent(tmp_path)
    body = _RunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
        }
    )
    barrier = Barrier(2)

    def start_capture() -> AgentExecutionRecorder | None:
        barrier.wait()
        return agent._start_agent_execution_capture(body)

    with ThreadPoolExecutor(max_workers=2) as pool:
        recorders = list(pool.map(lambda _: start_capture(), range(2)))

    owner = next(recorder for recorder in recorders if recorder is not None)
    assert sum(recorder is not None for recorder in recorders) == 1
    asyncio.run(agent._finish_agent_execution_capture(owner))

    capture = AgentExecutionCaptureStore(tmp_path).read("3-4")
    assert capture is not None
    assert capture.capture_complete is False
    assert capture.warnings == ["duplicate_agent_execution_capture_owner"]


def test_capture_write_failure_does_not_change_run_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _capture_agent(tmp_path)

    def fail_write(_self, _capture) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(AgentExecutionCaptureStore, "write", fail_write)

    response = TestClient(agent.setup_webserver()).post(
        "/run",
        json={
            "responses_create_params": {"input": []},
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_slow_capture_flush_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _capture_agent(tmp_path)
    recorder = AgentExecutionRecorder("3-4", "agent", EXACT_COVERAGE)
    flush_started = Event()
    release_flush = Event()
    original_write = AgentExecutionCaptureStore.write

    def blocking_write(store: AgentExecutionCaptureStore, capture: AgentExecutionCapture) -> None:
        flush_started.set()
        assert release_flush.wait(timeout=5)
        original_write(store, capture)

    monkeypatch.setattr(AgentExecutionCaptureStore, "write", blocking_write)
    flush_task = asyncio.create_task(agent._finish_agent_execution_capture(recorder))
    while not flush_started.is_set():
        await asyncio.sleep(0)

    heartbeat_ran = False

    async def heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0)
        heartbeat_ran = True

    await heartbeat()
    assert heartbeat_ran
    assert not flush_task.done()

    release_flush.set()
    await flush_task
    assert AgentExecutionCaptureStore(tmp_path).read("3-4") is not None


def test_store_handles_many_independent_rollout_writers(tmp_path: Path) -> None:
    store = AgentExecutionCaptureStore(tmp_path)

    def write(index: int) -> None:
        store.write(AgentExecutionRecorder(f"{index}-0", "agent", EXACT_COVERAGE).capture())

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(128)))

    assert {store.read(f"{index}-0").rollout_id for index in range(128)} == {f"{index}-0" for index in range(128)}
