# Configuration

For basic configuration instructions, see [this documentation](https://developers.openai.com/codex/config-basic).

For advanced configuration instructions, see [this documentation](https://developers.openai.com/codex/config-advanced).

For a full configuration reference, see [this documentation](https://developers.openai.com/codex/config-reference).

## Connecting to MCP servers

Codex can connect to MCP servers configured in `~/.codex/config.toml`. See the configuration reference for the latest MCP server options:

- https://developers.openai.com/codex/config-reference

## Apps (Connectors)

Use `$` in the composer to insert a ChatGPT connector; the popover lists accessible
apps. The `/apps` command lists available and installed apps. Connected apps appear first
and are labeled as connected; others are marked as can be installed.

## Notify

Codex can run a notification hook when the agent finishes a turn. See the configuration reference for the latest notification settings:

- https://developers.openai.com/codex/config-reference

## Native RLM

On first run, Codex writes missing `[native_rlm]` defaults into
`~/.codex/config.toml`. Native RLM can then be configured directly there:

```toml
[native_rlm]
enabled = true
max_iterations = 20
max_llm_calls = 50
llm_batch_concurrency = 8
max_output_chars = 10000
exec_timeout_ms = 180000
python_command = "python3"
verbose = true
sub_model = "gpt-5-mini"  # model for llm_query sub-calls (default: gpt-5-mini)
```

`CODEX_NATIVE_RLM*` environment variables are still supported and take precedence over
config values when both are present.

## JSON Schema

The generated JSON Schema for `config.toml` lives at `codex-rs/core/config.schema.json`.

## Notices

Codex stores "do not show again" flags for some UI prompts under the `[notice]` table.

Ctrl+C/Ctrl+D quitting uses a ~1 second double-press hint (`ctrl + c again to quit`).
