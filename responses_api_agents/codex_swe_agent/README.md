# Codex SWE Agent (sandbox-bound)

Runs OpenAI **Codex** (`codex exec --json`) **inside a Gym sandbox** as a custom /
blackbox agent, instead of as a host subprocess. It demonstrates the
sandbox-bound interceptor path end to end:

1. `run()` starts a sandbox (e.g. ECS Fargate) from the task image.
2. A per-rollout **capture proxy** is started and bound to the rollout's
   `session_id`; the in-box `OPENAI_BASE_URL` is pointed at it, so every model
   call is recorded (with token-ids when the policy is the Gym model server).
3. Codex runs in the box; the workspace diff (`git diff`) is the result patch.
4. The trajectory is assembled from the capture store (`assemble_trajectory`),
   carrying per-turn `generation_token_ids` for RL.
5. Verify reuses the `swe_agents` SWE-bench parser (`parse_and_check_tests`)
   on in-box test output.

## What changes vs `codex_agent`

`codex_agent` runs `codex` on the **host** and points its base-URL at the model.
This agent re-hosts it **in a sandbox** and points the base-URL at the
**capture proxy** — gaining isolation plus trajectory/token-id capture.

## Quick start

Configure a model server and a sandbox provider, then:

```bash
ng_run "+config_paths=[responses_api_agents/codex_swe_agent/configs/codex_swe_agent.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

ng_collect_rollouts \
  +agent_name=codex_swe_agent \
  +input_jsonl_fpath=responses_api_agents/codex_swe_agent/data/example.jsonl \
  +output_jsonl_fpath=codex_swe_rollouts.jsonl \
  +limit=1
```

The task row carries the SWE-bench `instance_id`, `fail_to_pass`, `pass_to_pass`
and `test_framework` in `responses_create_params.metadata`; set `eval_command`
in the config (or per task) to run the instance's tests in-box.

## Notes

- The `codex` CLI is installed **in the sandbox** (`npm i -g @openai/codex`) or
  baked into the task image — it is not a host dependency.
- For ECS, the box reaches the proxy through the provider's egress (SSH reverse
  tunnel); set `proxy_advertise_url` to the address the box should use.
- RL token-ids require the Gym model server (`return_token_id_information`); a
  third-party endpoint yields an eval-only trajectory.
