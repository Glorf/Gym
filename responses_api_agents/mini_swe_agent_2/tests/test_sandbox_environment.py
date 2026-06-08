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

from responses_api_agents.mini_swe_agent_2.sandbox_environment import MiniSWESandboxEnvironment, Submitted


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
    from responses_api_agents.mini_swe_agent_2.sandbox_environment import (
        MiniSWESandboxEnvironmentConfig,
    )

    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig.__new__(MiniSWESandboxEnvironmentConfig)
    env.config.activate_conda = activate_conda
    env.config.conda_env = conda_env
    return env


def test_command_passthrough_when_conda_disabled() -> None:
    env = _env_with(activate_conda=False, conda_env="testbed")
    assert env._command("git apply patch.diff", "/testbed") == "git apply patch.diff"


def test_command_resolves_conda_without_relying_on_path() -> None:
    # The exec shell is non-login (conda not on PATH), so the wrapper must not
    # depend solely on `conda info --base`; it sources conda.sh from known roots
    # and must not use an `&&` chain that aborts when a root is missing.
    env = _env_with(activate_conda=True, conda_env="testbed")
    wrapped = env._command("git apply patch.diff", "/testbed")

    assert wrapped.startswith("cd /testbed && ")
    assert "/opt/miniconda3/etc/profile.d/conda.sh" not in wrapped  # built dynamically, not hardcoded mid-string
    assert "/opt/miniconda3" in wrapped  # but /opt/miniconda3 is one of the search roots
    assert "conda activate testbed && git apply patch.diff" in wrapped
    # The sourcing loop runs in a group so a missing root does not abort the command.
    assert wrapped.count("&&") >= 2
    assert "for __base in" in wrapped
