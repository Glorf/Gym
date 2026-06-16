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
"""Claude Code as a thin :class:`SandboxCliAgent` subclass, runnable against any backend.

Runs ``claude -p --output-format stream-json`` inside a Gym sandbox with
``model_api="messages"``, so the base routes it through the ``translate_anthropic``
interceptor (Anthropic Messages <-> OpenAI Chat) and Claude Code can run against
an OpenAI-compatible backend. This module only knows how to launch claude and
parse its Anthropic stream-json.
"""

from __future__ import annotations

import json
from typing import Any, Optional
import shlex
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.sandbox_cli_agent import LaunchPlan, SandboxCliAgent, SandboxCliAgentConfig


DEFAULT_CLAUDE_INSTALL = (
    "set -e; command -v curl >/dev/null 2>&1 || "
    "(apt-get update -qq && apt-get install -y -qq curl xz-utils >/dev/null 2>&1) || true; "
    "if [ ! -x /opt/nodejs/bin/node ]; then mkdir -p /opt/nodejs && "
    "curl -fsSL https://nodejs.org/dist/v22.15.0/node-v22.15.0-linux-x64.tar.xz "
    "| tar -xJ --strip-components=1 -C /opt/nodejs; fi; "
    "export PATH=/opt/nodejs/bin:$PATH; npm install -g @anthropic-ai/claude-code@latest"
)


def _extract_text(content: list[Any]) -> str:
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _extract_thinking(content: list[Any]) -> str:
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") in ("thinking", "reasoning"):
            parts.append(b.get("thinking") or b.get("text") or "")
    return "\n".join(p for p in parts if p)


def parse_stream_json(stdout: str) -> tuple[list[Any], dict]:
    """Convert ``claude -p --output-format=stream-json`` stdout into (output_items, usage)."""
    raw_events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    output_items: list[Any] = []
    pending_calls: dict[str, dict] = {}
    buffered_think: str | None = None
    total_input = 0
    total_output = 0

    for event in raw_events:
        etype = event.get("type")
        if etype == "result":
            usage = event.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
        elif etype == "assistant":
            message = event.get("message", {})
            content = message.get("content") or []
            usage = message.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
            if not isinstance(content, list):
                content = []
            think = _extract_thinking(content)
            if think:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think
            text = _extract_text(content)
            if text:
                if buffered_think:
                    text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                    buffered_think = None
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                input_data = block.get("input") or {}
                arguments = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                pending_calls[call_id] = {"name": block.get("name", ""), "call_id": call_id, "arguments": arguments}
        elif etype == "user":
            message = event.get("message", {})
            content = message.get("content") or []
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                call_info = pending_calls.pop(tool_id, None)
                if call_info:
                    output_items.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=call_info["arguments"],
                            call_id=tool_id,
                            name=call_info["name"],
                            type="function_call",
                            id=tool_id,
                            status="completed",
                        )
                    )
                result_content = block.get("content") or ""
                result_text = _extract_text(result_content) if isinstance(result_content, list) else str(result_content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_id,
                        output=result_text,
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": total_input, "output_tokens": total_output}


def claude_settings_json() -> str:
    """Minimal Claude settings: telemetry/attribution off."""
    return json.dumps(
        {
            "env": {
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
    )


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


class ClaudeCodeSweAgentConfig(SandboxCliAgentConfig):
    model: str = "claude-sonnet-4-5"
    model_api: str = "messages"
    max_turns: int = 30
    claude_install_command: str = DEFAULT_CLAUDE_INSTALL
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None


class ClaudeCodeSweAgent(SandboxCliAgent):
    config: ClaudeCodeSweAgentConfig

    def build_launch(self, *, box_base_url, prompt, system_prompt, workdir, config_dir) -> LaunchPlan:
        setup = [
            f"mkdir -p {shlex.quote(config_dir)}",
            f"cat > {shlex.quote(config_dir + '/settings.json')} <<'EOF'\n{claude_settings_json()}\nEOF",
        ]
        cmd = claude_command(
            prompt=prompt,
            model=self.config.model,
            max_turns=self.config.max_turns,
            system_prompt=system_prompt,
            allowed_tools=self.config.allowed_tools,
            disallowed_tools=self.config.disallowed_tools,
        )
        env = {
            "ANTHROPIC_API_KEY": "dummy-key",  # pragma: allowlist secret — proxy injects the real key
            "ANTHROPIC_AUTH_TOKEN": "local",  # pragma: allowlist secret
            "ANTHROPIC_BASE_URL": box_base_url,
            "ANTHROPIC_MODEL": self.config.model,
            "IS_SANDBOX": "1",
            "CLAUDE_CONFIG_DIR": config_dir,
        }
        return LaunchPlan(
            run_command=cmd,
            env=env,
            setup_commands=setup,
            install_command=self.config.claude_install_command,
            path_prepend=self.config.node_bin_dir,
        )

    def parse_stdout(self, stdout: str) -> list[Any]:
        items, _usage = parse_stream_json(stdout)
        return items


if __name__ == "__main__":
    ClaudeCodeSweAgent.run_webserver()
