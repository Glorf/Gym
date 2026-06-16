# Claude Code SWE Agent (sandbox-bound, any backend)

Runs **Claude Code** (`claude -p --output-format stream-json`) **inside a Gym
sandbox**, and — crucially — against **any backend**, not just the Anthropic API.

Claude Code speaks the Anthropic Messages API (`/v1/messages`). This harness
points `ANTHROPIC_BASE_URL` at a per-rollout **capture proxy** running in
`translate_anthropic` mode, which:

1. translates Anthropic `/v1/messages` requests to **OpenAI Chat Completions**
   for the Gym/vLLM model server (the "any backend" requirement);
2. records the (OpenAI-shaped) exchange with token-ids into the capture store;
3. translates the response back to Anthropic so Claude Code is none the wiser.

The workspace diff (`git diff`) is the result patch; the trajectory is assembled
from the capture store (token-ids for RL); verify reuses the `swe_agents`
SWE-bench parser in-box.

## Quick start

```bash
ng_run "+config_paths=[responses_api_agents/claude_code_swe_agent/configs/claude_code_swe_agent.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  +agent_name=claude_code_swe_agent \
  +input_jsonl_fpath=responses_api_agents/claude_code_swe_agent/data/example.jsonl \
  +output_jsonl_fpath=claude_swe_rollouts.jsonl \
  +limit=1
```

## Notes

- The `claude` CLI is installed **in the sandbox**
  (`npm i -g @anthropic-ai/claude-code`) or baked into the task image.
- **Streaming caveat:** Claude Code streams by default; the proxy currently
  buffers, so the translation forces a non-streaming upstream call. SSE
  translation is the remaining piece for full streaming fidelity.
- RL token-ids require the Gym model server (`return_token_id_information`).
