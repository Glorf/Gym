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
"""Claude Code as a sandbox-bound custom agent, runnable against *any* backend.

Claude Code speaks the Anthropic Messages API. This harness runs it **inside a
Gym sandbox** with ``ANTHROPIC_BASE_URL`` pointed at a per-rollout capture proxy
running in ``translate_anthropic`` mode, so Anthropic ``/v1/messages`` calls are
translated to OpenAI Chat Completions for the Gym/vLLM model server (the "any
backend" requirement) while the trajectory and token-ids are captured uniformly.

``run()`` owns the lifecycle; verify reuses the SWE-bench parser in-box.

NOTE: Claude Code streams by default; the proxy currently buffers (translation
forces a non-streaming upstream call). SSE translation is the remaining piece for
full streaming fidelity.
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
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec

# Reuse the existing Claude stream-json parser and the shared SWE helpers.
from responses_api_agents.claude_code_agent.app import parse_stream_json
from responses_api_agents.codex_swe_agent.app import extract_instruction, swebench_reward


LOG = logging.getLogger(__name__)


def claude_command(
    *,
    prompt: str,
    model: str,
    max_turns: int,
    system_prompt: Optional[str],
    allowed_tools: Optional[str],
    disallowed_tools: Optional[str],
) -> str:
    """Shell command to run ``claude -p --output-format stream-json`` in the box."""
    parts = [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--bare",
        "--max-turns",
        str(max_turns),
        "--model",
        shlex.quote(model),
    ]
    if system_prompt:
        parts += ["--append-system-prompt", shlex.quote(system_prompt)]
    if allowed_tools:
        parts += ["--allowedTools", shlex.quote(allowed_tools)]
    if disallowed_tools:
        parts += ["--disallowedTools", shlex.quote(disallowed_tools)]
    parts += ["--", shlex.quote(prompt)]
    return " ".join(parts)


def claude_settings_json() -> str:
    """Minimal Claude settings: telemetry/attribution off (matches the host agent)."""
    return json.dumps(
        {
            "env": {
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
    )


class ClaudeCodeSweAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None
    model_base_url: Optional[str] = None
    model: str = "claude-sonnet-4-5"
    concurrency: int = 8
    timeout_s: int = 1800
    max_turns: int = 30

    sandbox: dict[str, Any]
    image: Optional[str] = None
    image_template: Optional[str] = None
    workdir: str = "/testbed"

    install_claude_in_box: bool = True
    node_bin_dir: str = "/opt/nodejs/bin"
    claude_install_command: str = (
        "set -e; command -v curl >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq curl xz-utils >/dev/null 2>&1) || true; "
        "if [ ! -x /opt/nodejs/bin/node ]; then mkdir -p /opt/nodejs && "
        "curl -fsSL https://nodejs.org/dist/v22.15.0/node-v22.15.0-linux-x64.tar.xz "
        "| tar -xJ --strip-components=1 -C /opt/nodejs; fi; "
        "export PATH=/opt/nodejs/bin:$PATH; npm install -g @anthropic-ai/claude-code@latest"
    )
    model_api_key: str = ""  # pragma: allowlist secret — real upstream key (proxy injects it)
    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None

    capture_dir: str = "outputs/claude_code_swe_agent/captures"
    proxy_host: str = "127.0.0.1"
    proxy_advertise_url: Optional[str] = None
    return_token_ids: bool = True

    eval_command: Optional[str] = None


class ClaudeCodeSweAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSweAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    patch_exists: bool = False


class ClaudeCodeSweAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeSweAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _resolve_base_url(self) -> str:
        # Model ROOT (no /v1): the proxy forwards the agent's full path onto it.
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
            # Claude appends /v1/messages, so the endpoint URL has no /v1; the
            # translate interceptor rewrites it to /v1/chat/completions upstream.
            provider_options={
                "outside_endpoints": [{"url": proxy.handle.url, "env_var": "ANTHROPIC_BASE_URL"}]
            },
        )

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError(
            "ClaudeCodeSweAgent runs the full sandbox lifecycle in run(); it has no standalone responses() episode."
        )

    async def _gather_trajectory(self, session_id: str, claude_stdout: str) -> tuple[list[Any], bool]:
        """Prefer the captured (OpenAI-shaped) trajectory carrying generation_token_ids;
        fall back to Claude's Anthropic stream-json when nothing was captured."""
        exchanges = CaptureStore(self.config.capture_dir).read(session_id)
        captured = assemble_trajectory(exchanges)
        if captured:
            return captured, has_token_ids(captured)
        parsed, _usage = parse_stream_json(claude_stdout)
        return parsed, False

    async def run(
        self,
        request: Request,
        body: ClaudeCodeSweAgentRunRequest = Body(),
    ) -> ClaudeCodeSweAgentVerifyResponse:
        params = body.responses_create_params
        metadata = dict(getattr(params, "metadata", None) or {})
        body_input = getattr(params, "input", None)
        if isinstance(body_input, str):
            body_input = [NeMoGymEasyInputMessage(role="user", content=body_input)]
        user_message, input_system = extract_instruction(body_input)
        system_prompt = "\n\n".join(p for p in [self.config.system_prompt, input_system] if p) or None

        session_id = f"claude-{uuid4().hex[:12]}"
        config_dir = f"{self.config.workdir.rstrip('/')}/.claude"
        base_url = self._resolve_base_url()
        inject = {"return_token_id_information": True} if self.config.return_token_ids else {}

        # translate_anthropic: Claude's /v1/messages -> OpenAI chat for any backend.
        proxy = start_capture_proxy(
            model_base_url=base_url,
            session_id=session_id,
            store_dir=self.config.capture_dir,
            host=self.config.proxy_host,
            advertise_url=self.config.proxy_advertise_url,
            inject_extra_body=inject,
            upstream_api_key=self.config.model_api_key or None,
            request_timeout=float(self.config.timeout_s),
            translate_anthropic=True,
            translate_model_override=self.config.model,
        )

        sandbox = AsyncSandbox(self.config.sandbox, self._sandbox_spec(metadata, proxy), delete_on_stop=True)
        claude_stdout = ""
        patch = ""
        test_output = ""
        try:
            await sandbox.start()

            box_base_url = sandbox.resolved_endpoint_url("ANTHROPIC_BASE_URL") or proxy.sandbox_base_url
            path_prefix = f"export PATH={shlex.quote(self.config.node_bin_dir)}:$PATH"
            box_env = {
                "ANTHROPIC_API_KEY": "dummy-key",  # pragma: allowlist secret — proxy injects the real key
                "ANTHROPIC_AUTH_TOKEN": "local",  # pragma: allowlist secret
                "ANTHROPIC_BASE_URL": box_base_url,
                "ANTHROPIC_MODEL": self.config.model,
                "IS_SANDBOX": "1",
                "CLAUDE_CONFIG_DIR": config_dir,
            }
            await sandbox.exec(f"mkdir -p {shlex.quote(config_dir)}")
            await sandbox.exec(
                f"cat > {shlex.quote(config_dir + '/settings.json')} <<'EOF'\n{claude_settings_json()}\nEOF"
            )
            if self.config.install_claude_in_box:
                install = await sandbox.exec(self.config.claude_install_command, timeout_s=self.config.timeout_s)
                if install.return_code != 0:
                    LOG.warning("claude install in box returned %s: %s", install.return_code, install.stderr)

            cmd = claude_command(
                prompt=user_message,
                model=self.config.model,
                max_turns=self.config.max_turns,
                system_prompt=system_prompt,
                allowed_tools=self.config.allowed_tools,
                disallowed_tools=self.config.disallowed_tools,
            )
            result = await sandbox.exec(
                f"{path_prefix} && {cmd}", cwd=self.config.workdir, env=box_env, timeout_s=self.config.timeout_s
            )
            claude_stdout = result.stdout or ""

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

        output_items, rl_ready = await self._gather_trajectory(session_id, claude_stdout)
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

        return ClaudeCodeSweAgentVerifyResponse(
            responses_create_params=params,
            response=gym_resp,
            reward=reward,
            turns_used=turns,
            patch_exists=bool(patch.strip()),
            **verify_fields,
        )


if __name__ == "__main__":
    ClaudeCodeSweAgent.run_webserver()
