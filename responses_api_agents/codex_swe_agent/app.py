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
"""Codex as a sandbox-bound custom agent.

Unlike ``codex_agent`` (which runs ``codex exec`` as a *host* subprocess), this
harness runs Codex **inside a Gym sandbox**: it starts a box, stands up a
per-rollout capture proxy, points Codex's ``base_url`` at that proxy so every
model call is recorded with token-ids, runs ``codex exec --json`` via
``sandbox.exec``, collects the patch with ``git diff``, assembles the trajectory
from the capture store, and grades it with the reused SWE-bench parser.

``run()`` owns the whole lifecycle; ``responses()`` is intentionally not a
standalone episode here.
"""

from __future__ import annotations

import json
import logging
import shlex
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body, Request
from pydantic import ConfigDict

from nemo_gym.adapters.capture_store import CaptureStore, assemble_trajectory, has_token_ids
from nemo_gym.adapters.sandbox_capture import start_capture_proxy
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec
from responses_api_agents.swe_agents.swe_bench_ext.utils import parse_and_check_tests


LOG = logging.getLogger(__name__)

# ``codex exec --json`` item types that carry a tool action.
_TOOL_ITEM_TYPES = {"command_execution", "file_change", "mcp_tool_call", "web_search", "patch_apply"}


def parse_codex_jsonl(stdout: str) -> tuple[list[Any], dict]:
    """Convert ``codex exec --json`` stdout (JSONL events) into (output_items, usage)."""
    output_items: list[Any] = []
    buffered_reasoning: str | None = None
    total_input = 0
    total_output = 0

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        etype = event.get("type")
        if etype in ("turn.completed", "turn.failed"):
            usage = event.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
            continue
        if etype != "item.completed":
            continue

        item = event.get("item") or {}
        if not isinstance(item, dict):
            continue
        itype = item.get("type") or item.get("item_type")

        if itype == "reasoning":
            text = item.get("text") or ""
            if text:
                buffered_reasoning = (buffered_reasoning + "\n" + text) if buffered_reasoning else text
        elif itype in ("agent_message", "assistant_message"):
            text = item.get("text") or ""
            if buffered_reasoning:
                text = f"<think>\n{buffered_reasoning}\n</think>\n\n{text}"
                buffered_reasoning = None
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif itype in _TOOL_ITEM_TYPES:
            call_id = item.get("id") or f"call-{uuid4().hex[:8]}"
            name = "shell" if itype == "command_execution" else itype
            if itype == "command_execution":
                arguments = json.dumps({"command": item.get("command", "")})
            else:
                arguments = json.dumps({k: v for k, v in item.items() if k not in ("id", "type", "item_type")})
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=name,
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            output = item.get("aggregated_output") or item.get("output") or ""
            exit_code = item.get("exit_code")
            if exit_code is not None:
                output = f"{output}\n(exit_code={exit_code})"
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(output),
                    status="completed",
                )
            )

    return output_items, {"input_tokens": total_input, "output_tokens": total_output}


def extract_instruction(body_input: Any) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    if isinstance(body_input, str):
        return body_input, None
    items = list(body_input or [])
    system_message: Optional[str] = None

    def _text(content: Any) -> str:
        if isinstance(content, list):
            return "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        return content or ""

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            system_message = _text(content)
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            user_message = _text(content)
            break
    return user_message, system_message


def codex_config_toml(*, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY") -> str:
    """Codex ``config.toml`` routing the Responses wire API at our proxy.

    codex >= 0.14 dropped ``wire_api = "chat"``; it speaks the OpenAI Responses
    API, so the proxy forwards ``/v1/responses`` to the backend.
    """
    return (
        f'model = "{model}"\n'
        'model_provider = "gym"\n'
        "[model_providers.gym]\n"
        'name = "gym"\n'
        f'base_url = "{base_url}"\n'
        f'env_key = "{api_key_env}"\n'
        'wire_api = "responses"\n'
    )


def codex_command(*, prompt: str, model: str, sandbox_mode: str, skip_git_repo_check: bool, codex_home: str) -> str:
    """Shell command to run ``codex exec --json`` inside the box."""
    parts = [f"CODEX_HOME={shlex.quote(codex_home)}", "codex", "exec", "--json", "--model", shlex.quote(model)]
    if skip_git_repo_check:
        parts.append("--skip-git-repo-check")
    if sandbox_mode:
        parts += ["--sandbox", shlex.quote(sandbox_mode)]
    parts += ["--", shlex.quote(prompt)]
    return " ".join(parts)


def swebench_reward(test_output: str, metadata: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Grade SWE-bench resolution from in-box test output via the reused parser."""
    fail_to_pass = metadata.get("fail_to_pass") or metadata.get("FAIL_TO_PASS") or []
    pass_to_pass = metadata.get("pass_to_pass") or metadata.get("PASS_TO_PASS") or []
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    if isinstance(pass_to_pass, str):
        pass_to_pass = json.loads(pass_to_pass)
    framework = metadata.get("test_framework") or "pytest"

    report = parse_and_check_tests(
        test_output=test_output,
        test_framework=framework,
        fail_to_pass=list(fail_to_pass),
        pass_to_pass=list(pass_to_pass),
        instance_id=str(metadata.get("instance_id") or ""),
    )
    return (1.0 if report.get("resolved") else 0.0), report


class CodexSweAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None
    model_base_url: Optional[str] = None
    model: str = "gpt-5-codex"
    concurrency: int = 8
    timeout_s: int = 1800

    # sandbox
    sandbox: dict[str, Any]  # provider config, e.g. {"name": "ecs_fargate", ...}
    image: Optional[str] = None
    image_template: Optional[str] = None  # e.g. "swebench/sweb.eval.x86_64.{instance_id}:latest"
    workdir: str = "/workspace"

    # codex in-box. The default bootstraps a static Node (image-agnostic, no root
    # pkg manager needed beyond curl/xz) then installs codex globally under it.
    install_codex_in_box: bool = True
    node_bin_dir: str = "/opt/nodejs/bin"
    codex_install_command: str = (
        "set -e; command -v curl >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq curl xz-utils >/dev/null 2>&1) || true; "
        "if [ ! -x /opt/nodejs/bin/node ]; then mkdir -p /opt/nodejs && "
        "curl -fsSL https://nodejs.org/dist/v22.15.0/node-v22.15.0-linux-x64.tar.xz "
        "| tar -xJ --strip-components=1 -C /opt/nodejs; fi; "
        "export PATH=/opt/nodejs/bin:$PATH; npm install -g @openai/codex@latest"
    )
    model_api_key: str = ""  # pragma: allowlist secret — real upstream key (proxy injects it)
    codex_sandbox_mode: str = "workspace-write"
    skip_git_repo_check: bool = True
    system_prompt: Optional[str] = None

    # capture proxy
    capture_dir: str = "outputs/codex_swe_agent/captures"
    proxy_host: str = "127.0.0.1"
    proxy_advertise_url: Optional[str] = None  # URL the box uses (tunnelled egress)
    return_token_ids: bool = True

    # verify
    eval_command: Optional[str] = None  # in-box test command; None => verify skipped (reward 0)


class CodexSweAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexSweAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    patch_exists: bool = False


class CodexSweAgent(SimpleResponsesAPIAgent):
    config: CodexSweAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _resolve_base_url(self) -> str:
        # Model ROOT (no trailing /v1): the proxy forwards the agent's full path
        # (e.g. /v1/chat/completions) onto this, so /v1 must not be doubled.
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(cfg)
        return (self.config.model_base_url or "").rstrip("/")

    def _sandbox_spec(self, metadata: dict[str, Any], proxy: Any) -> SandboxSpec:
        image = self.config.image
        if self.config.image_template and metadata.get("instance_id"):
            image = self.config.image_template.format(instance_id=metadata["instance_id"])
        return SandboxSpec(
            image=image,
            workdir=self.config.workdir,
            timeout_s=self.config.timeout_s,
            metadata={"instance_id": str(metadata.get("instance_id") or "")},
            # Reverse-tunnel the harness-side capture proxy into the box; the agent
            # reads its in-box address back via sandbox.resolved_endpoint_url().
            provider_options={
                "outside_endpoints": [{"url": proxy.handle.url + "/v1", "env_var": "OPENAI_BASE_URL"}]
            },
        )

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError(
            "CodexSweAgent runs the full sandbox lifecycle in run(); it has no standalone responses() episode."
        )

    async def _gather_trajectory(self, session_id: str, codex_stdout: str) -> tuple[list[Any], bool]:
        """Prefer the captured model trajectory; fall back to Codex's own JSONL.

        The captured trajectory is assembled into ``NeMoGymResponseOutputMessageForTraining``
        items carrying per-assistant ``generation_token_ids`` (RL-usable) when the
        policy is the Gym model server; ``has_token_ids`` reports whether they are present.
        """
        store = CaptureStore(self.config.capture_dir)
        exchanges = store.read(session_id)
        captured = assemble_trajectory(exchanges)
        if captured:
            return captured, has_token_ids(captured)
        parsed, _usage = parse_codex_jsonl(codex_stdout)
        return parsed, False

    async def run(
        self,
        request: Request,
        body: CodexSweAgentRunRequest = Body(),
    ) -> CodexSweAgentVerifyResponse:
        params = body.responses_create_params
        metadata = dict(getattr(params, "metadata", None) or {})
        body_input = getattr(params, "input", None)
        if isinstance(body_input, str):
            body_input = [NeMoGymEasyInputMessage(role="user", content=body_input)]
        user_message, input_system = extract_instruction(body_input)
        system_prompt = "\n\n".join(p for p in [self.config.system_prompt, input_system] if p) or None
        prompt = f"{system_prompt}\n\n---\n\n{user_message}" if system_prompt else user_message

        session_id = f"codex-{uuid4().hex[:12]}"
        codex_home = f"{self.config.workdir.rstrip('/')}/.codex"
        base_url = self._resolve_base_url()
        inject = {"return_token_id_information": True} if self.config.return_token_ids else {}

        proxy = start_capture_proxy(
            model_base_url=base_url,
            session_id=session_id,
            store_dir=self.config.capture_dir,
            host=self.config.proxy_host,
            advertise_url=self.config.proxy_advertise_url,
            inject_extra_body=inject,
            upstream_api_key=self.config.model_api_key or None,
            request_timeout=float(self.config.timeout_s),
        )

        sandbox = AsyncSandbox(self.config.sandbox, self._sandbox_spec(metadata, proxy), delete_on_stop=True)
        codex_stdout = ""
        patch = ""
        test_output = ""
        try:
            await sandbox.start()

            # In-box address of the capture proxy via the provider reverse tunnel;
            # falls back to the harness URL for same-host providers.
            box_base_url = sandbox.resolved_endpoint_url("OPENAI_BASE_URL") or proxy.sandbox_base_url
            path_prefix = f"export PATH={shlex.quote(self.config.node_bin_dir)}:$PATH"
            box_env = {
                "OPENAI_API_KEY": "dummy-key",  # pragma: allowlist secret — proxy injects the real key
                "CODEX_HOME": codex_home,
            }
            await sandbox.exec(f"mkdir -p {shlex.quote(codex_home)}")
            await sandbox.exec(
                f"cat > {shlex.quote(codex_home + '/config.toml')} <<'EOF'\n"
                f"{codex_config_toml(base_url=box_base_url, model=self.config.model)}EOF"
            )
            if self.config.install_codex_in_box:
                install = await sandbox.exec(self.config.codex_install_command, timeout_s=self.config.timeout_s)
                if install.return_code != 0:
                    LOG.warning("codex install in box returned %s: %s", install.return_code, install.stderr)

            cmd = codex_command(
                prompt=prompt,
                model=self.config.model,
                sandbox_mode=self.config.codex_sandbox_mode,
                skip_git_repo_check=self.config.skip_git_repo_check,
                codex_home=codex_home,
            )
            result = await sandbox.exec(
                f"{path_prefix} && {cmd}", cwd=self.config.workdir, env=box_env, timeout_s=self.config.timeout_s
            )
            codex_stdout = result.stdout or ""

            patch_res = await sandbox.exec(
                f"cd {shlex.quote(self.config.workdir)} && git add -A && git diff --cached",
                timeout_s=300,
            )
            patch = patch_res.stdout or ""

            if self.config.eval_command:
                eval_res = await sandbox.exec(self.config.eval_command, cwd=self.config.workdir, timeout_s=self.config.timeout_s)
                test_output = (eval_res.stdout or "") + "\n" + (eval_res.stderr or "")
        finally:
            proxy.stop()
            await sandbox.stop()

        output_items, rl_ready = await self._gather_trajectory(session_id, codex_stdout)
        turns = sum(
            1
            for item in output_items
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
        )

        reward = 0.0
        verify_fields: dict[str, Any] = {}
        if self.config.eval_command and test_output.strip():
            reward, verify_fields = swebench_reward(test_output, metadata)

        gym_resp = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=output_items,
            tool_choice=getattr(params, "tool_choice", None),
            tools=getattr(params, "tools", None),
            parallel_tool_calls=getattr(params, "parallel_tool_calls", None),
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
            metadata={
                "instance_id": str(metadata.get("instance_id") or ""),
                "patch": patch,
                "session_id": session_id,
                "rl_token_ids": str(rl_ready).lower(),
            }
            | {k: str(v) for k, v in verify_fields.items()},
        )

        return CodexSweAgentVerifyResponse(
            responses_create_params=params,
            response=gym_resp,
            reward=reward,
            turns_used=turns,
            patch_exists=bool(patch.strip()),
            **verify_fields,
        )


if __name__ == "__main__":
    CodexSweAgent.run_webserver()
