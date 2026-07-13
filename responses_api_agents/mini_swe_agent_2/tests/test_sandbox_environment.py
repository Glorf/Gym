# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from nemo_gym.agent_execution_capture import AgentExecutionCoverage, AgentExecutionRecorder
from responses_api_agents.mini_swe_agent_2.sandbox_environment import (
    MiniSWESandboxEnvironment,
    MiniSWESandboxEnvironmentConfig,
    Submitted,
)


def test_check_finished_raises_submitted_for_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    try:
        env._check_finished(
            {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
                "returncode": 0,
                "exception_info": "",
            }
        )
    except Submitted as error:
        assert error.messages == (
            {
                "role": "exit",
                "content": "patch contents\n",
                "extra": {"exit_status": "Submitted", "submission": "patch contents\n"},
            },
        )
    else:
        raise AssertionError("Expected Submitted")


def test_check_finished_ignores_nonzero_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    env._check_finished(
        {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
            "returncode": 1,
            "exception_info": "",
        }
    )


def _env_with(activate_conda: bool, conda_env):
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig.__new__(MiniSWESandboxEnvironmentConfig)
    env.config.activate_conda = activate_conda
    env.config.conda_env = conda_env
    return env


def test_command_passthrough_when_conda_disabled() -> None:
    env = _env_with(activate_conda=False, conda_env="testbed")
    assert env._command("git apply patch.diff") == "git apply patch.diff"


def test_command_resolves_conda_without_relying_on_path() -> None:
    # Non-login ECS exec shell: conda isn't on PATH, so source conda.sh from known roots
    # via a grouped loop (not an `&&` chain that aborts on a missing root). cwd is passed
    # to exec(cwd=...), so the command itself must not `cd`.
    env = _env_with(activate_conda=True, conda_env="testbed")
    wrapped = env._command("git apply patch.diff")

    assert "for __base in" in wrapped
    assert "/opt/miniconda3" in wrapped  # one of the search roots
    assert "conda activate testbed && git apply patch.diff" in wrapped
    assert wrapped.count("&&") >= 2
    assert not wrapped.startswith("cd ")  # cwd handled by exec(cwd=...), not via cd


def test_execute_passes_configured_cwd_to_exec() -> None:
    class FakeSandbox:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def exec(self, command: str, **kwargs: Any):
            self.calls.append({"command": command, **kwargs})
            return type("Result", (), {"stdout": "ok", "stderr": None, "return_code": 0})()

    fake_sandbox = FakeSandbox()
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig(
        image="image:tag",
        provider={"fake": {}},
        cwd="/default",
        activate_conda=False,
    )
    env._sandbox = fake_sandbox
    env._agent_execution_recorder = None
    env._capture_agent_commands = False

    assert env.execute("pwd", cwd="/repo") == {"output": "ok", "returncode": 0, "exception_info": ""}
    assert fake_sandbox.calls[-1]["command"] == "pwd"
    assert fake_sandbox.calls[-1]["cwd"] == "/repo"

    env.config.activate_conda = True
    env.config.conda_env = "testbed"
    env.execute("python -V", cwd="/repo")
    cmd = fake_sandbox.calls[-1]["command"]
    assert "for __base in" in cmd and "conda activate testbed && python -V" in cmd
    assert "cd /repo" not in cmd
    assert fake_sandbox.calls[-1]["cwd"] == "/repo"


def test_execute_records_only_agent_selected_commands() -> None:
    class FakeSandbox:
        def exec(self, _command: str, **_kwargs: Any):
            return type("Result", (), {"stdout": "ok", "stderr": None, "return_code": 0})()

    ticks = iter([100, 250])
    recorder = AgentExecutionRecorder(
        "2-1",
        "mini_swe_agent_2",
        AgentExecutionCoverage(lineage="exact", model_call_attribution="exact", tool_timing="exact"),
        clock=lambda: next(ticks),
        clock_id="mini-swe-worker-clock",
    )
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig(
        image="image:tag",
        provider={"fake": {}},
        cwd="/workspace",
    )
    env._sandbox = FakeSandbox()
    env._capture_agent_commands = False
    env.set_agent_execution_recorder(recorder)

    env.execute({"command": "setup", "tool_call_id": "setup-call"})
    with env.capture_agent_commands():
        env.execute({"command": "agent", "tool_call_id": "call-1"})
        env.execute({"command": "grade", "tool_call_id": "grade-call"}, is_eval=True)
    env.execute({"command": "cleanup", "tool_call_id": "cleanup-call"})

    capture = recorder.capture()
    assert len(capture.tool_spans) == 1
    span = capture.tool_spans[0]
    assert span.tool_call_ids == ["call-1"]
    assert span.granularity == "individual"
    assert span.measurement_scope == "caller_round_trip"
    assert span.clock_id == "mini-swe-worker-clock"
    assert (span.started_ns, span.ended_ns, span.status) == (100, 250, "returned")


def test_execute_without_active_capture_does_not_read_the_clock() -> None:
    class FakeSandbox:
        def exec(self, _command: str, **_kwargs: Any):
            return type("Result", (), {"stdout": "ok", "stderr": None, "return_code": 0})()

    def unexpected_clock_read() -> int:
        raise AssertionError("capture-disabled execution must not read the recorder clock")

    recorder = AgentExecutionRecorder(
        "2-1",
        "mini_swe_agent_2",
        AgentExecutionCoverage(lineage="exact", model_call_attribution="exact", tool_timing="exact"),
        clock=unexpected_clock_read,
    )
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig(image="image:tag", provider={"fake": {}})
    env._sandbox = FakeSandbox()
    env._capture_agent_commands = False
    env.set_agent_execution_recorder(recorder)

    env.execute({"command": "setup", "tool_call_id": "setup-call"})

    assert recorder.capture().tool_spans == []
