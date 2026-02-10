use crate::client::ModelClientSession;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::tools::ToolSpec;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::config::Config;
use crate::config::types::NativeRlmToml;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::protocol::EventMsg;
use crate::protocol::TurnDiffEvent;
use crate::protocol::WarningEvent;
use crate::stream_events_utils::response_input_to_response_item;
use crate::tools::ToolRouter;
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::context::ToolPayload;
use crate::tools::parallel::ToolCallRuntime;
use crate::tools::router::ToolCall;
use crate::tools::spec::JsonSchema;
use crate::turn_diff_tracker::TurnDiffTracker;
use codex_async_utils::OrCancelExt;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::ShellToolCallParams;
use codex_protocol::user_input::UserInput;
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const DEFAULT_MAX_ITERATIONS: u32 = 20;
const DEFAULT_MAX_LLM_CALLS: u32 = 50;
const DEFAULT_LLM_BATCH_CONCURRENCY: usize = 8;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 100_000;
const DEFAULT_EXEC_TIMEOUT_MS: u64 = 180_000;
const DEFAULT_PYTHON_COMMAND: &str = "python3";
const PYTHON_REPL_TIMEOUT_ERROR_PREFIX: &str = "native_rlm Python REPL timed out after";
const REPL_HISTORY_PROMPT_MAX_OUTPUT_CHARS: usize = 5_000;

const NATIVE_RLM_ENABLED_ENV: &str = "CODEX_NATIVE_RLM";
const NATIVE_RLM_MAX_ITERATIONS_ENV: &str = "CODEX_NATIVE_RLM_MAX_ITERATIONS";
const NATIVE_RLM_MAX_LLM_CALLS_ENV: &str = "CODEX_NATIVE_RLM_MAX_LLM_CALLS";
const NATIVE_RLM_LLM_BATCH_CONCURRENCY_ENV: &str = "CODEX_NATIVE_RLM_LLM_BATCH_CONCURRENCY";
const NATIVE_RLM_MAX_OUTPUT_CHARS_ENV: &str = "CODEX_NATIVE_RLM_MAX_OUTPUT_CHARS";
const NATIVE_RLM_EXEC_TIMEOUT_MS_ENV: &str = "CODEX_NATIVE_RLM_EXEC_TIMEOUT_MS";
const NATIVE_RLM_PYTHON_COMMAND_ENV: &str = "CODEX_NATIVE_RLM_PYTHON_COMMAND";
const NATIVE_RLM_VERBOSE_ENV: &str = "CODEX_NATIVE_RLM_VERBOSE";

const ACTION_INSTRUCTIONS: &str = r#"
You are operating in native recursive language model (RLM) mode.
RLM means: iterate in a persistent Python REPL, reason over context, use tools for outside-world actions, then finish with SUBMIT.
You are also the Codex coding assistant for this conversation.
Your objective is to complete the user's latest unresolved request.

Critical interpretation:
- Only `conversation_history` entries are conversation turns.
- This controller prompt is protocol guidance, not user content.
- Prioritize newest turns (tail-first), not earliest turns.
- Tool outputs are authoritative. A claim is true only if tool output in this run proves it.

Built-ins:
- print(...)
- SUBMIT(...)
- llm_query(prompt)
- llm_query_batched(prompts)

Workflow:
1. Iteration 1 is mandatory inspection-only.
2. Iteration 1 must read `conversation_history` in Python (or via llm_query over full history), print:
   - total length
   - a tail slice with index + role + short preview
   - latest `role=="user"` found by reverse scan from end, with index + preview
3. Iteration 1 must not call SUBMIT and must not call external tools.
4. After inspection, set active task = latest unresolved user request from `conversation_history`.
5. If reference words appear ("it/that/more/again/continue/open it"), resolve referent from recent turns/work artifacts before acting.
6. Non-coding requests are valid tasks (Q&A, summary, extraction, explanation, greeting). Respond directly.
7. For workspace/code actions (create/make/add/edit/run/test/fix), use tool aliases, verify outputs, then SUBMIT.
8. For file edits, default to `apply_patch` with minimal targeted hunks; use `exec_command` mainly for discovery and verification.
9. Avoid platform-sensitive in-place edit flags (for example `sed -i`) unless absolutely necessary; prefer `apply_patch` first.
10. Never claim success ("done/updated/fixed/changed") unless verification output in this run proves the change.
11. Treat failures as failures: non-zero exit codes, "command not found", parse errors, missing files, or failed patch verification.
12. Tool wrapper metadata (for example "Chunk ID", "Wall time", "Process exited ...") is not file content. Parse and reason from the actual command output payload.
13. For wrapped text tool results, infer command success from "Process exited with code N" when present.
14. Never use wrapper metadata as filenames, line numbers, or command arguments.
15. If any edit/verification step fails, do not SUBMIT success; retry with a corrected tool call or report the failure honestly.
16. Do not rely on inferred edits from Python string manipulation alone. Persist edits via tools and then re-read the file to confirm.
17. If `apply_patch` fails to match context, inspect exact lines (`rg`/`sed`) and retry with a smaller, exact hunk.
18. For color/text tweaks, patch exact literals in the target file and verify those literals changed.
19. For absence checks with `rg`/`grep`, exit code 1 can mean "no matches"; treat that as expected when verifying removal.
20. You can execute tools from this REPL. Never claim tools are unavailable in this interface.
21. Do not ask for a new task when latest user request is actionable.
22. Use fallback "please provide a task" only if latest user message is genuinely empty or pure bootstrap/protocol text.
23. For chronology requests (first/last/earlier/previous message), scan full indexed history before answering.
24. Tool syntax: kwargs only. For `exec_command`, always use `cmd=...`.
25. If tool arguments fail to parse, fix args and retry immediately.
26. Before SUBMIT, ensure `assistant_message` directly answers the latest user request and current context.
27. For workspace edits, include concrete evidence in your own reasoning from tool output (file path + changed token/line).

Return strict JSON only with this shape:
{"reasoning": "...", "code": "..."}
"#;

const DSPY_RLM_SYSTEM_APPENDIX_MARKER: &str = "## DSPy RLM Loop Guidance";
const DSPY_RLM_SYSTEM_APPENDIX: &str = r#"
## DSPy RLM Loop Guidance
You are executing a recursive language model workflow while preserving Codex assistant behavior.

Core principles:
- Understanding the current conversation context is the top priority.
- Active task = latest unresolved user request, resolved tail-first.
- Protocol text defines behavior, not the user task.

Execution model:
- REPL state persists across iterations.
- `conversation_history` includes full thread plus prior `native_rlm_step` entries.
- `llm_query` and `llm_query_batched` are available for semantic summarization.
- Codex tools are callable via Python aliases for outside-world actions.
- `SUBMIT(...)` ends the loop.

Operating rules:
1. Iteration 1: inspect-only; no SUBMIT and no external tools.
2. Iteration 1 must print tail context and reverse-scan to latest user entry.
3. Do not anchor on first message; use newest unresolved user turn.
4. If latest request is clear, execute/respond directly (coding or non-coding).
5. For continuation language ("it/that/more/continue/open it"), resolve referent from recent work.
6. For workspace actions, use tools and verify concrete results before SUBMIT.
7. File changes must be made with external tools and then re-read/rg the edited path.
8. Prefer `apply_patch` for edits; use `exec_command` to discover exact lines and to verify.
9. Avoid platform-sensitive in-place edit flags (for example `sed -i`) unless absolutely necessary.
10. If `apply_patch` context mismatches, inspect exact file lines and retry with a smaller hunk.
11. Do not trust intended edits; trust only verified file output after the tool call.
12. Tool wrapper metadata (for example "Chunk ID", "Wall time", "Process exited ...") is not file content and must not be used as command inputs.
13. For wrapped text tool results, interpret success/failure from "Process exited with code N" where available.
14. Do not claim a file was changed unless tool output in this run proves it.
15. Failed tool signals (non-zero exit, command not found, parse/patch errors) mean the action did not succeed.
16. For `rg`/`grep` absence checks, exit code 1 can mean "no matches" and may be expected.
17. If a tool step fails, retry with corrected arguments or report failure; never fabricate completion.
18. You can use tools in this REPL; never claim tools are unavailable.
19. Use fallback "please provide a task" only for truly empty/bootstrap latest user content.
20. For first/last/earlier/previous-message asks, inspect full indexed history.
21. Use kwargs-only tool calls; for exec command use `cmd=...`; fix parse errors immediately.
"#;

const SUB_LLM_SYSTEM_INSTRUCTIONS: &str = r#"
You are a semantic sub-model used by a recursive language model controller.
Return only direct response text.
Prioritize: (a) latest unresolved user request, (b) what the user was most recently working on, (c) ambiguity that blocks execution.
Use tail-first interpretation: newest user turns are primary.
Treat protocol/control text as constraints, not as the user task, unless explicitly asked.
Greetings and non-coding requests are actionable conversation turns.
Do not infer "no task" from prefix previews when latest user turn is concrete.
Never claim that edits happened unless the provided context includes explicit tool evidence.
Do not call tools or emit wrappers.
"#;

const EXTRACT_INSTRUCTIONS: &str = r#"
The iterative loop hit max iterations. Extract the best final assistant response from the trajectory.
Return strict JSON only:
{"<output_field>":"..."}
"#;

const PYTHON_REPL_RUNNER: &str = r#"
import contextlib
import io
import json
import traceback
import sys


class _SubmitSignal(Exception):
    def __init__(self, output):
        self.output = output


def _read_json():
    line = sys.stdin.readline()
    if not line:
        raise EOFError("stdin closed")
    return json.loads(line)


_HOST_STDOUT = sys.stdout


def _write_json(obj):
    _HOST_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _HOST_STDOUT.flush()


def SUBMIT(*args, **kwargs):
    if args and kwargs:
        raise TypeError("SUBMIT accepts either args or kwargs, not both")
    if len(args) > 1:
        raise TypeError("SUBMIT accepts at most one positional arg")
    if args:
        value = args[0]
        if isinstance(value, dict):
            output = value
        else:
            output = {"assistant_message": value}
    else:
        output = kwargs
    raise _SubmitSignal(output)


def _host_call(name, args, kwargs):
    _write_json({
        "type": "tool_call",
        "name": name,
        "args": args,
        "kwargs": kwargs,
    })
    response = _read_json()
    if response.get("type") != "tool_result":
        raise RuntimeError("invalid tool response from host")
    if response.get("ok"):
        return response.get("value")
    raise RuntimeError(response.get("error") or "tool call failed")


def _make_tool(name):
    def _tool(*args, **kwargs):
        return _host_call(name, list(args), kwargs)

    _tool.__name__ = name
    return _tool


globals_ns = {"SUBMIT": SUBMIT}

init = _read_json()
if init.get("type") != "init":
    raise RuntimeError("first message must be init")

for tool_name in init.get("tools", []):
    globals_ns[tool_name] = _make_tool(tool_name)

while True:
    try:
        request = _read_json()
    except EOFError:
        break

    if request.get("type") != "exec":
        _write_json({
            "type": "exec_result",
            "kind": "error",
            "error": "invalid request type",
            "stdout": "",
        })
        continue

    variables = request.get("variables") or {}
    if isinstance(variables, dict):
        globals_ns.update(variables)

    code = request.get("code", "")
    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, globals_ns, globals_ns)
        _write_json({
            "type": "exec_result",
            "kind": "output",
            "output": stdout_buffer.getvalue(),
            "stdout": stdout_buffer.getvalue(),
        })
    except _SubmitSignal as final:
        _write_json({
            "type": "exec_result",
            "kind": "final",
            "output": final.output,
            "stdout": stdout_buffer.getvalue(),
        })
    except Exception as e:
        _write_json({
            "type": "exec_result",
            "kind": "error",
            "error": f"{type(e).__name__}: {e}",
            "stdout": stdout_buffer.getvalue(),
            "traceback": traceback.format_exc(),
        })
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeRlmSettings {
    enabled: bool,
    max_iterations: u32,
    max_llm_calls: u32,
    llm_batch_concurrency: usize,
    max_output_chars: usize,
    exec_timeout_ms: u64,
    python_command: String,
    verbose: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct NativeRlmEnvOverrides {
    enabled: Option<bool>,
    max_iterations: Option<u32>,
    max_llm_calls: Option<u32>,
    llm_batch_concurrency: Option<usize>,
    max_output_chars: Option<usize>,
    exec_timeout_ms: Option<u64>,
    python_command: Option<String>,
    verbose: Option<bool>,
}

impl Default for NativeRlmSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_llm_calls: DEFAULT_MAX_LLM_CALLS,
            llm_batch_concurrency: DEFAULT_LLM_BATCH_CONCURRENCY,
            max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            exec_timeout_ms: DEFAULT_EXEC_TIMEOUT_MS,
            python_command: DEFAULT_PYTHON_COMMAND.to_string(),
            verbose: false,
        }
    }
}

impl NativeRlmSettings {
    pub(crate) fn from_config(config: &Config) -> Self {
        let mut settings = Self::from_config_values(Some(&config.native_rlm));
        settings.apply_env_overrides(NativeRlmEnvOverrides::from_env());
        settings
    }

    fn from_config_values(native_rlm: Option<&NativeRlmToml>) -> Self {
        let command = native_rlm
            .and_then(|cfg| cfg.python_command.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_PYTHON_COMMAND)
            .to_string();

        Self {
            enabled: native_rlm.and_then(|cfg| cfg.enabled).unwrap_or(false),
            max_iterations: native_rlm
                .and_then(|cfg| cfg.max_iterations)
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_MAX_ITERATIONS),
            max_llm_calls: native_rlm
                .and_then(|cfg| cfg.max_llm_calls)
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_MAX_LLM_CALLS),
            llm_batch_concurrency: native_rlm
                .and_then(|cfg| cfg.llm_batch_concurrency)
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_LLM_BATCH_CONCURRENCY),
            max_output_chars: native_rlm
                .and_then(|cfg| cfg.max_output_chars)
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS),
            exec_timeout_ms: native_rlm
                .and_then(|cfg| cfg.exec_timeout_ms)
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_EXEC_TIMEOUT_MS),
            python_command: command,
            verbose: native_rlm.and_then(|cfg| cfg.verbose).unwrap_or(false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn from_raw_values(
        enabled: Option<&str>,
        max_iterations: Option<&str>,
        max_llm_calls: Option<&str>,
        llm_batch_concurrency: Option<&str>,
        max_output_chars: Option<&str>,
        exec_timeout_ms: Option<&str>,
        python_command: Option<&str>,
        verbose: Option<&str>,
    ) -> Self {
        let command = python_command
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_PYTHON_COMMAND)
            .to_string();

        Self {
            enabled: parse_bool(enabled).unwrap_or(false),
            max_iterations: parse_positive_u32(max_iterations).unwrap_or(DEFAULT_MAX_ITERATIONS),
            max_llm_calls: parse_positive_u32(max_llm_calls).unwrap_or(DEFAULT_MAX_LLM_CALLS),
            llm_batch_concurrency: parse_positive_usize(llm_batch_concurrency)
                .unwrap_or(DEFAULT_LLM_BATCH_CONCURRENCY),
            max_output_chars: parse_positive_usize(max_output_chars)
                .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS),
            exec_timeout_ms: parse_positive_u64(exec_timeout_ms).unwrap_or(DEFAULT_EXEC_TIMEOUT_MS),
            python_command: command,
            verbose: parse_bool(verbose).unwrap_or(false),
        }
    }

    fn apply_env_overrides(&mut self, overrides: NativeRlmEnvOverrides) {
        let NativeRlmEnvOverrides {
            enabled,
            max_iterations,
            max_llm_calls,
            llm_batch_concurrency,
            max_output_chars,
            exec_timeout_ms,
            python_command,
            verbose,
        } = overrides;

        if let Some(enabled) = enabled {
            self.enabled = enabled;
        }
        if let Some(max_iterations) = max_iterations {
            self.max_iterations = max_iterations;
        }
        if let Some(max_llm_calls) = max_llm_calls {
            self.max_llm_calls = max_llm_calls;
        }
        if let Some(llm_batch_concurrency) = llm_batch_concurrency {
            self.llm_batch_concurrency = llm_batch_concurrency;
        }
        if let Some(max_output_chars) = max_output_chars {
            self.max_output_chars = max_output_chars;
        }
        if let Some(exec_timeout_ms) = exec_timeout_ms {
            self.exec_timeout_ms = exec_timeout_ms;
        }
        if let Some(python_command) = python_command {
            self.python_command = python_command;
        }
        if let Some(verbose) = verbose {
            self.verbose = verbose;
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn status_message(&self) -> Option<String> {
        if !self.enabled {
            return None;
        }

        Some(format!(
            "Native RLM mode enabled:\n  max_iterations={}\n  max_llm_calls={}\n  llm_batch_concurrency={}\n  max_output_chars={}\n  exec_timeout_ms={}\n  python_command={}",
            self.max_iterations,
            self.max_llm_calls,
            self.llm_batch_concurrency,
            self.max_output_chars,
            self.exec_timeout_ms,
            self.python_command
        ))
    }
}

impl NativeRlmEnvOverrides {
    fn from_env() -> Self {
        let enabled = std::env::var(NATIVE_RLM_ENABLED_ENV).ok();
        let max_iterations = std::env::var(NATIVE_RLM_MAX_ITERATIONS_ENV).ok();
        let max_llm_calls = std::env::var(NATIVE_RLM_MAX_LLM_CALLS_ENV).ok();
        let llm_batch_concurrency = std::env::var(NATIVE_RLM_LLM_BATCH_CONCURRENCY_ENV).ok();
        let max_output_chars = std::env::var(NATIVE_RLM_MAX_OUTPUT_CHARS_ENV).ok();
        let exec_timeout_ms = std::env::var(NATIVE_RLM_EXEC_TIMEOUT_MS_ENV).ok();
        let python_command = std::env::var(NATIVE_RLM_PYTHON_COMMAND_ENV).ok();
        let verbose = std::env::var(NATIVE_RLM_VERBOSE_ENV).ok();

        Self {
            enabled: parse_bool(enabled.as_deref()),
            max_iterations: parse_positive_u32(max_iterations.as_deref()),
            max_llm_calls: parse_positive_u32(max_llm_calls.as_deref()),
            llm_batch_concurrency: parse_positive_usize(llm_batch_concurrency.as_deref()),
            max_output_chars: parse_positive_usize(max_output_chars.as_deref()),
            exec_timeout_ms: parse_positive_u64(exec_timeout_ms.as_deref()),
            python_command: python_command
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            verbose: parse_bool(verbose.as_deref()),
        }
    }
}

pub(crate) fn should_disable_history_truncation(config: &Config) -> bool {
    NativeRlmSettings::from_config(config).enabled()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ActionStep {
    reasoning: String,
    code: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalOutputSpec {
    schema_driven: bool,
    output_schema: Value,
    field_names: Vec<String>,
    required_fields: Vec<String>,
    field_schemas: HashMap<String, Value>,
    enforce_no_additional_properties: bool,
}

impl FinalOutputSpec {
    fn default_assistant_message() -> Self {
        let output_schema = json!({
            "type": "object",
            "properties": {
                "assistant_message": { "type": "string" }
            },
            "required": ["assistant_message"],
            "additionalProperties": false
        });

        let field_names = vec!["assistant_message".to_string()];
        let required_fields = field_names.clone();
        let field_schemas =
            HashMap::from([("assistant_message".to_string(), json!({"type": "string"}))]);

        Self {
            schema_driven: false,
            output_schema,
            field_names,
            required_fields,
            field_schemas,
            enforce_no_additional_properties: true,
        }
    }

    fn from_turn_schema(schema: Option<&Value>) -> Self {
        let Some(Value::Object(root)) = schema else {
            return Self::default_assistant_message();
        };

        let Some(Value::Object(properties)) = root.get("properties") else {
            return Self::default_assistant_message();
        };

        if properties.is_empty() {
            return Self::default_assistant_message();
        }

        let mut field_schemas: HashMap<String, Value> = properties
            .iter()
            .map(|(name, property_schema)| (name.clone(), property_schema.clone()))
            .collect();

        let mut field_names = field_schemas.keys().cloned().collect::<Vec<String>>();
        field_names.sort();

        let mut required_fields = root
            .get("required")
            .and_then(Value::as_array)
            .map(|required| {
                required
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        required_fields.sort();
        required_fields.dedup();

        for required in &required_fields {
            if !field_schemas.contains_key(required) {
                field_schemas.insert(required.clone(), Value::Object(Map::new()));
                field_names.push(required.clone());
            }
        }
        field_names.sort();
        field_names.dedup();

        let enforce_no_additional_properties =
            matches!(root.get("additionalProperties"), Some(Value::Bool(false)));

        Self {
            schema_driven: true,
            output_schema: Value::Object(root.clone()),
            field_names,
            required_fields,
            field_schemas,
            enforce_no_additional_properties,
        }
    }

    fn submit_signature_hint(&self) -> String {
        self.field_names
            .iter()
            .map(|name| format!("{name}=..."))
            .collect::<Vec<String>>()
            .join(", ")
    }

    fn output_fields_description(&self) -> String {
        self.field_names
            .iter()
            .map(|name| {
                let type_name = self
                    .field_schemas
                    .get(name)
                    .and_then(schema_type_description)
                    .unwrap_or_else(|| "any".to_string());
                let required = if self.required_fields.contains(name) {
                    "required"
                } else {
                    "optional"
                };
                format!("- `{name}` ({required}, type: {type_name})")
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReplVariable {
    name: String,
    type_name: String,
    total_length: usize,
    preview: String,
}

impl ReplVariable {
    fn from_json(name: String, value: &Value, preview_chars: usize) -> Self {
        let value_text = if matches!(value, Value::Array(_) | Value::Object(_)) {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        } else {
            value.to_string()
        };
        let is_truncated = value_text.chars().count() > preview_chars;
        let preview = if is_truncated {
            value_text.chars().take(preview_chars).collect::<String>() + "..."
        } else {
            value_text.clone()
        };

        Self {
            name,
            type_name: json_type_name(value).to_string(),
            total_length: value_text.chars().count(),
            preview,
        }
    }

    fn format(&self) -> String {
        format!(
            "Variable: `{}`\nType: {}\nTotal length: {} chars\nPreview:\n```\n{}\n```",
            self.name, self.type_name, self.total_length, self.preview
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReplEntry {
    reasoning: String,
    code: String,
    output: String,
}

impl ReplEntry {
    fn format(&self, index: usize, max_output_chars: usize) -> String {
        let output = if self.output.chars().count() > max_output_chars {
            let truncated = self
                .output
                .chars()
                .take(max_output_chars)
                .collect::<String>();
            format!(
                "{truncated}\n... (truncated to {max_output_chars}/{})",
                self.output.chars().count()
            )
        } else {
            self.output.clone()
        };
        let mut formatted = String::new();
        let _ = writeln!(formatted, "=== Step {} ===", index + 1);
        if !self.reasoning.is_empty() {
            let _ = writeln!(formatted, "Reasoning: {}", self.reasoning);
        }
        let _ = writeln!(formatted, "Code:\n```python\n{}\n```", self.code);
        let _ = writeln!(
            formatted,
            "Output ({} chars):\n{}",
            self.output.chars().count(),
            output
        );
        formatted
    }
}

#[derive(Debug, Clone)]
enum ToolBindingKind {
    Function,
    Custom,
    LocalShell,
    Mcp { server: String, tool: String },
}

#[derive(Debug, Clone)]
struct ToolBinding {
    alias: String,
    tool_name: String,
    signature: String,
    description: String,
    kind: ToolBindingKind,
}

#[derive(Debug, Clone)]
struct ToolCatalog {
    bindings_by_alias: HashMap<String, ToolBinding>,
    aliases: Vec<String>,
    documentation: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RunnerEvent {
    ToolCall {
        name: String,
        #[serde(default)]
        args: Vec<Value>,
        #[serde(default)]
        kwargs: Map<String, Value>,
    },
    ExecResult {
        kind: String,
        #[serde(default)]
        output: Option<Value>,
        #[serde(default)]
        stdout: String,
        #[serde(default)]
        error: String,
        #[serde(default)]
        traceback: String,
    },
}

#[derive(Debug)]
enum InterpreterResult {
    Output {
        stdout: String,
    },
    Final {
        output: Value,
        stdout: String,
    },
    Error {
        error: String,
        stdout: String,
        traceback: String,
    },
}

#[derive(Debug)]
struct PythonRepl {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl PythonRepl {
    async fn start(settings: &NativeRlmSettings, tool_aliases: &[String]) -> CodexResult<Self> {
        let command_parts = shlex::split(&settings.python_command)
            .filter(|parts| !parts.is_empty())
            .unwrap_or_else(|| vec![settings.python_command.clone()]);
        let mut command = Command::new(&command_parts[0]);
        command.args(command_parts.iter().skip(1));
        command
            .arg("-u")
            .arg("-c")
            .arg(PYTHON_REPL_RUNNER)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn().map_err(|err| {
            CodexErr::Fatal(format!(
                "failed to spawn python interpreter `{}`: {err}",
                settings.python_command
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CodexErr::Fatal("failed to capture python stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CodexErr::Fatal("failed to capture python stdout".to_string()))?;

        let mut repl = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };

        let init = json!({
            "type": "init",
            "tools": tool_aliases,
        });
        repl.send_json(&init).await?;
        Ok(repl)
    }

    async fn send_json<T: Serialize>(&mut self, value: &T) -> CodexResult<()> {
        let mut line = serde_json::to_vec(value)?;
        line.push(b'\n');
        self.stdin.write_all(&line).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_event(&mut self) -> CodexResult<RunnerEvent> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line).await?;
        if read == 0 {
            return Err(CodexErr::Fatal(
                "python interpreter exited unexpectedly".to_string(),
            ));
        }
        let event: RunnerEvent = serde_json::from_str(line.trim()).map_err(|err| {
            CodexErr::Fatal(format!(
                "invalid message from python interpreter: {err}; raw={}",
                line.trim()
            ))
        })?;
        Ok(event)
    }

    async fn begin_exec(&mut self, code: &str, variables: &Map<String, Value>) -> CodexResult<()> {
        let request = json!({
            "type": "exec",
            "code": code,
            "variables": variables,
        });
        self.send_json(&request).await
    }

    async fn send_tool_result(&mut self, result: Result<Value, String>) -> CodexResult<()> {
        match result {
            Ok(value) => {
                let payload = json!({
                    "type": "tool_result",
                    "ok": true,
                    "value": value,
                });
                self.send_json(&payload).await
            }
            Err(error) => {
                let payload = json!({
                    "type": "tool_result",
                    "ok": false,
                    "error": error,
                });
                self.send_json(&payload).await
            }
        }
    }

    async fn shutdown(&mut self) {
        let _ = self.stdin.shutdown().await;
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

impl Drop for PythonRepl {
    fn drop(&mut self) {
        self.child.start_kill().ok();
    }
}

struct NativeRlmRunner<'a> {
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    client_session: &'a mut ModelClientSession,
    turn_metadata_header: Option<&'a str>,
    cancellation_token: CancellationToken,
    settings: NativeRlmSettings,
    rlm_base_instructions: BaseInstructions,
    sub_llm_base_instructions: BaseInstructions,
    final_output_spec: FinalOutputSpec,
    tool_runtime: ToolCallRuntime,
    tool_catalog: ToolCatalog,
    llm_calls_used: u32,
    tracker: SharedTurnDiffTracker,
}

impl<'a> NativeRlmRunner<'a> {
    async fn build_iteration_variables(
        &self,
        history: &[ReplEntry],
    ) -> CodexResult<(Map<String, Value>, Vec<String>)> {
        let history_for_prompt = self.sess.clone_history().await.for_prompt();
        let variables = build_variables_map(&history_for_prompt, history)?;
        let variable_infos = variables
            .iter()
            .map(|(name, value)| ReplVariable::from_json(name.clone(), value, 500).format())
            .collect::<Vec<String>>();
        Ok((variables, variable_infos))
    }

    async fn run(mut self, _input: &[UserInput]) -> CodexResult<Option<String>> {
        if let Some(status) = self.settings.status_message() {
            self.sess
                .send_event(
                    &self.turn_context,
                    EventMsg::Warning(WarningEvent { message: status }),
                )
                .await;
        }

        let mut repl = PythonRepl::start(&self.settings, &self.tool_catalog.aliases).await?;
        let mut history: Vec<ReplEntry> = Vec::new();
        let mut initial_history_inspection_completed = false;

        for iteration in 0..self.settings.max_iterations {
            let (variables, variable_infos) =
                self.build_iteration_variables(history.as_slice()).await?;
            let conversation_history = variables
                .get("conversation_history")
                .cloned()
                .unwrap_or(Value::Null);
            let latest_user_message = latest_user_message_text(&conversation_history);
            let conversation_tail = conversation_tail_summary(&conversation_history, 8, 180);
            let likely_workspace_request = latest_user_message
                .as_deref()
                .map(is_likely_workspace_edit_request)
                .unwrap_or(false);
            let action = self
                .generate_action(
                    &variable_infos,
                    &history,
                    iteration,
                    latest_user_message.as_deref(),
                    &conversation_tail,
                )
                .await?;
            let code = strip_code_fences(&action.code);
            let code_uses_external_tools =
                code_invokes_external_tool_alias(&code, &self.tool_catalog.aliases);
            if self.settings.verbose {
                self.sess
                    .notify_background_event(
                        &self.turn_context,
                        format!(
                            "native_rlm iteration {}/{}",
                            iteration + 1,
                            self.settings.max_iterations
                        ),
                    )
                    .await;
                self.emit_iteration_plan_debug(iteration, &action.reasoning, &code)
                    .await;
            }

            if !initial_history_inspection_completed
                && !iteration_one_code_has_required_history_inspection(&code)
            {
                let guidance = "[Policy Error] Initial inspection is required before finalizing. Print `conversation_history` length/tail/latest-user preview or use `llm_query` over conversation history, then continue.";
                history.push(ReplEntry {
                    reasoning: action.reasoning,
                    code,
                    output: format_output(guidance, self.settings.max_output_chars),
                });
                continue;
            }
            if !initial_history_inspection_completed {
                initial_history_inspection_completed = true;
            }

            let result = self.execute_code(&mut repl, &code, &variables).await?;
            if self.settings.verbose {
                self.emit_iteration_result_debug(iteration, &result).await;
            }
            let restart_repl = Self::should_restart_repl_after_result(&result);
            match self
                .process_iteration_result(
                    action,
                    code,
                    result,
                    &latest_user_message,
                    likely_workspace_request,
                    code_uses_external_tools,
                    &mut history,
                )
                .await
            {
                IterationOutcome::Continue => {
                    if restart_repl {
                        self.restart_repl_after_timeout(&mut repl).await?;
                    }
                }
                IterationOutcome::Final(message) => {
                    repl.shutdown().await;
                    self.emit_turn_diff_if_any().await;
                    self.emit_final_assistant_message(&message).await;
                    return Ok(Some(message));
                }
            }
        }

        repl.shutdown().await;
        self.sess
            .send_event(
                &self.turn_context,
                EventMsg::Warning(WarningEvent {
                    message: format!(
                        "Native RLM hit max_iterations={} and is extracting a final answer from trajectory.",
                        self.settings.max_iterations
                    ),
                }),
            )
            .await;

        let (_, variable_infos) = self.build_iteration_variables(history.as_slice()).await?;
        let extracted = self.extract_fallback(&variable_infos, &history).await?;
        self.emit_turn_diff_if_any().await;
        self.emit_final_assistant_message(&extracted).await;
        Ok(Some(extracted))
    }

    async fn generate_action(
        &mut self,
        variable_infos: &[String],
        history: &[ReplEntry],
        iteration: u32,
        latest_user_message: Option<&str>,
        conversation_tail: &str,
    ) -> CodexResult<ActionStep> {
        let prompt_text = render_action_prompt(
            &self.turn_context,
            variable_infos,
            history,
            &self.tool_catalog.documentation,
            &self.final_output_spec,
            iteration,
            self.settings.max_iterations,
            self.settings.max_llm_calls,
            latest_user_message,
            conversation_tail,
            &self.settings.python_command,
        );

        let output_schema = Some(json!({
            "type": "object",
            "properties": {
                "reasoning": { "type": "string" },
                "code": { "type": "string" }
            },
            "required": ["reasoning", "code"],
            "additionalProperties": false
        }));

        let text = self
            .query_model_text(
                &prompt_text,
                output_schema,
                self.rlm_base_instructions.clone(),
                self.turn_context.personality,
            )
            .await?;
        parse_action_step(&text)
    }

    async fn extract_fallback(
        &mut self,
        variable_infos: &[String],
        history: &[ReplEntry],
    ) -> CodexResult<String> {
        let prompt_text = render_extract_prompt(variable_infos, history, &self.final_output_spec);
        let output_schema = Some(self.final_output_spec.output_schema.clone());

        let text = self
            .query_model_text(
                &prompt_text,
                output_schema,
                self.rlm_base_instructions.clone(),
                self.turn_context.personality,
            )
            .await?;
        let extract_value = serde_json::from_str::<Value>(&text)
            .or_else(|_| {
                extract_json_object(&text)
                    .ok_or_else(|| {
                        CodexErr::Fatal(format!(
                            "native_rlm extract output was not valid JSON object: {text}"
                        ))
                    })
                    .and_then(|json_object| {
                        serde_json::from_str::<Value>(&json_object).map_err(CodexErr::from)
                    })
            })
            .map_err(|err| {
                CodexErr::Fatal(format!(
                    "native_rlm extract output was not valid JSON object: {err}; raw={text}"
                ))
            })?;
        let parsed =
            parse_final_output(extract_value, &self.final_output_spec).map_err(CodexErr::Fatal)?;
        Ok(final_output_to_assistant_message(
            &parsed,
            &self.final_output_spec,
        ))
    }

    async fn execute_code(
        &mut self,
        repl: &mut PythonRepl,
        code: &str,
        variables: &Map<String, Value>,
    ) -> CodexResult<InterpreterResult> {
        repl.begin_exec(code, variables).await?;
        let exec_timeout = Duration::from_millis(self.settings.exec_timeout_ms);

        loop {
            let event = match tokio::time::timeout(exec_timeout, repl.read_event()).await {
                Ok(event) => event?,
                Err(_) => {
                    return Ok(InterpreterResult::Error {
                        error: format!(
                            "native_rlm Python REPL timed out after {} ms while waiting for output",
                            self.settings.exec_timeout_ms
                        ),
                        stdout: String::new(),
                        traceback: String::new(),
                    });
                }
            };
            match event {
                RunnerEvent::ToolCall { name, args, kwargs } => {
                    if self.settings.verbose {
                        self.emit_verbose_warning(format!("native_rlm tool call start: {name}"))
                            .await;
                    }
                    let tool_result = match tokio::time::timeout(
                        exec_timeout,
                        self.handle_tool_call(name.clone(), args, kwargs),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => Err(format!(
                            "tool call `{name}` timed out after {} ms",
                            self.settings.exec_timeout_ms
                        )),
                    };
                    if self.settings.verbose {
                        let outcome = match &tool_result {
                            Ok(value) => {
                                format!(
                                    "ok: {}",
                                    serde_json::to_string(value)
                                        .unwrap_or_else(|_| "<non-serializable>".to_string())
                                )
                            }
                            Err(error) => format!("error: {error}"),
                        };
                        self.emit_verbose_warning(format!(
                            "native_rlm tool call done: {name} -> {outcome}"
                        ))
                        .await;
                    }
                    repl.send_tool_result(tool_result).await?;
                }
                RunnerEvent::ExecResult {
                    kind,
                    output,
                    stdout,
                    error,
                    traceback,
                } => {
                    return match kind.as_str() {
                        "output" => Ok(InterpreterResult::Output {
                            stdout: output
                                .and_then(|value| value.as_str().map(ToOwned::to_owned))
                                .unwrap_or(stdout),
                        }),
                        "final" => Ok(InterpreterResult::Final {
                            output: output.unwrap_or(Value::Null),
                            stdout,
                        }),
                        "error" => Ok(InterpreterResult::Error {
                            error,
                            stdout,
                            traceback,
                        }),
                        _ => Err(CodexErr::Fatal(format!(
                            "unknown exec result kind from python interpreter: {kind}"
                        ))),
                    };
                }
            }
        }
    }

    fn should_restart_repl_after_result(result: &InterpreterResult) -> bool {
        matches!(
            result,
            InterpreterResult::Error { error, .. }
                if is_python_repl_timeout_error(error)
        )
    }

    async fn restart_repl_after_timeout(&self, repl: &mut PythonRepl) -> CodexResult<()> {
        self.sess
            .send_event(
                &self.turn_context,
                EventMsg::Warning(WarningEvent {
                    message: format!(
                        "native_rlm restarting Python REPL after timeout (exec_timeout_ms={})",
                        self.settings.exec_timeout_ms
                    ),
                }),
            )
            .await;
        repl.shutdown().await;
        *repl = PythonRepl::start(&self.settings, &self.tool_catalog.aliases).await?;
        Ok(())
    }

    async fn emit_verbose_warning(&self, message: String) {
        self.sess
            .send_event(
                &self.turn_context,
                EventMsg::Warning(WarningEvent { message }),
            )
            .await;
    }

    async fn emit_iteration_plan_debug(&self, iteration: u32, reasoning: &str, code: &str) {
        let message = format!(
            "native_rlm iteration {}/{} plan\nreasoning:\n{}\n\ncode:\n```python\n{}\n```",
            iteration + 1,
            self.settings.max_iterations,
            reasoning,
            code
        );
        self.emit_verbose_warning(message).await;
    }

    async fn emit_iteration_result_debug(&self, iteration: u32, result: &InterpreterResult) {
        let result_text = match result {
            InterpreterResult::Output { stdout } => format!("output:\n{stdout}"),
            InterpreterResult::Final { output, stdout } => format!(
                "final output:\n{}\n\nstdout:\n{}",
                serde_json::to_string_pretty(output).unwrap_or_else(|_| output.to_string()),
                stdout
            ),
            InterpreterResult::Error {
                error,
                stdout,
                traceback,
            } => format!("error: {error}\n\nstdout:\n{stdout}\n\ntraceback:\n{traceback}"),
        };
        let message = format!(
            "native_rlm iteration {}/{} result\n{}",
            iteration + 1,
            self.settings.max_iterations,
            result_text
        );
        self.emit_verbose_warning(message).await;
    }

    async fn handle_tool_call(
        &mut self,
        name: String,
        args: Vec<Value>,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        match name.as_str() {
            "llm_query" => self.call_llm_query(args, kwargs).await,
            "llm_query_batched" => self.call_llm_query_batched(args, kwargs).await,
            _ => self.call_codex_tool(name, args, kwargs).await,
        }
    }

    async fn call_llm_query(
        &mut self,
        args: Vec<Value>,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let prompt = extract_prompt_argument(args, kwargs)?;
        if prompt.trim().is_empty() {
            return Err("llm_query prompt cannot be empty".to_string());
        }
        self.consume_llm_calls(1)?;
        self.run_sub_llm_prompt(&prompt)
            .await
            .map(Value::String)
            .map_err(|err| err.to_string())
    }

    async fn call_llm_query_batched(
        &mut self,
        args: Vec<Value>,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let prompts = extract_prompts_argument(args, kwargs)?;
        if prompts.iter().any(|prompt| prompt.trim().is_empty()) {
            return Err("llm_query_batched prompts cannot contain empty strings".to_string());
        }
        self.consume_llm_calls(prompts.len() as u32)?;
        if prompts.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }

        let sess = Arc::clone(&self.sess);
        let turn_context = Arc::clone(&self.turn_context);
        let cancellation_token = self.cancellation_token.child_token();
        let turn_metadata_header = self.turn_metadata_header.map(ToOwned::to_owned);
        let base_instructions = self.sub_llm_base_instructions.clone();
        let llm_batch_concurrency = self.settings.llm_batch_concurrency;

        let mut outputs = vec![Value::Null; prompts.len()];
        let results = futures::stream::iter(prompts.into_iter().enumerate())
            .map(|(index, prompt)| {
                let sess = Arc::clone(&sess);
                let turn_context = Arc::clone(&turn_context);
                let cancellation_token = cancellation_token.child_token();
                let turn_metadata_header = turn_metadata_header.clone();
                let base_instructions = base_instructions.clone();
                async move {
                    let mut client_session = sess.services.model_client.new_session();
                    let result = query_model_text_with_session(
                        sess.as_ref(),
                        turn_context.as_ref(),
                        &mut client_session,
                        turn_metadata_header.as_deref(),
                        &cancellation_token,
                        &prompt,
                        None,
                        base_instructions,
                        None,
                    )
                    .await;
                    (index, result)
                }
            })
            .buffer_unordered(llm_batch_concurrency)
            .collect::<Vec<(usize, CodexResult<String>)>>()
            .await;

        for (index, result) in results {
            outputs[index] = match result {
                Ok(text) => Value::String(text),
                Err(err) => Value::String(format!("[ERROR] {err}")),
            };
        }

        Ok(Value::Array(outputs))
    }

    fn consume_llm_calls(&mut self, amount: u32) -> Result<(), String> {
        if self.llm_calls_used.saturating_add(amount) > self.settings.max_llm_calls {
            return Err(format!(
                "LLM call limit exceeded: {} + {} > {}",
                self.llm_calls_used, amount, self.settings.max_llm_calls
            ));
        }
        self.llm_calls_used += amount;
        Ok(())
    }

    async fn run_sub_llm_prompt(&mut self, prompt: &str) -> CodexResult<String> {
        let mut client_session = self.sess.services.model_client.new_session();
        query_model_text_with_session(
            self.sess.as_ref(),
            self.turn_context.as_ref(),
            &mut client_session,
            self.turn_metadata_header,
            &self.cancellation_token,
            prompt,
            None,
            self.sub_llm_base_instructions.clone(),
            None,
        )
        .await
    }

    async fn call_codex_tool(
        &mut self,
        alias: String,
        args: Vec<Value>,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let binding = self
            .tool_catalog
            .bindings_by_alias
            .get(&alias)
            .cloned()
            .ok_or_else(|| format!("unknown tool alias: {alias}"))?;

        let (args, kwargs) = normalize_tool_call_arguments_for_native_rlm(
            &binding,
            args,
            kwargs,
            &self.settings.python_command,
        );

        let call_id = format!("native-rlm-{}", Uuid::new_v4());
        let payload = build_tool_payload(&binding, args, kwargs)?;
        let call = ToolCall {
            tool_name: binding.tool_name,
            call_id,
            payload,
        };
        let call_item = native_rlm_tool_call_to_response_item(&call);

        let response = self
            .tool_runtime
            .clone()
            .handle_tool_call(call, self.cancellation_token.child_token())
            .await
            .map_err(|err| err.to_string())?;

        if let Some(output_item) = response_input_to_response_item(&response) {
            self.sess
                .record_conversation_items(&self.turn_context, &[call_item, output_item])
                .await;
        }

        response_input_item_to_value(response)
    }

    async fn query_model_text(
        &mut self,
        user_prompt: &str,
        output_schema: Option<Value>,
        base_instructions: BaseInstructions,
        personality: Option<crate::config::types::Personality>,
    ) -> CodexResult<String> {
        query_model_text_with_session(
            self.sess.as_ref(),
            self.turn_context.as_ref(),
            self.client_session,
            self.turn_metadata_header,
            &self.cancellation_token,
            user_prompt,
            output_schema,
            base_instructions,
            personality,
        )
        .await
    }

    async fn process_iteration_result(
        &mut self,
        action: ActionStep,
        code: String,
        result: InterpreterResult,
        latest_user_message: &Option<String>,
        likely_workspace_request: bool,
        code_uses_external_tools: bool,
        history: &mut Vec<ReplEntry>,
    ) -> IterationOutcome {
        match result {
            InterpreterResult::Output { stdout } => {
                history.push(ReplEntry {
                    reasoning: action.reasoning,
                    code,
                    output: format_output(&stdout, self.settings.max_output_chars),
                });
                IterationOutcome::Continue
            }
            InterpreterResult::Error {
                error,
                stdout,
                traceback,
            } => {
                let mut combined = String::new();
                if !stdout.trim().is_empty() {
                    let _ = writeln!(combined, "{stdout}");
                }
                let _ = writeln!(combined, "[Error] {error}");
                if self.settings.verbose && !traceback.is_empty() {
                    let _ = writeln!(combined, "{traceback}");
                }
                history.push(ReplEntry {
                    reasoning: action.reasoning,
                    code,
                    output: format_output(&combined, self.settings.max_output_chars),
                });
                IterationOutcome::Continue
            }
            InterpreterResult::Final { output, stdout } => {
                let parsed = parse_final_output(output, &self.final_output_spec);
                match parsed {
                    Ok(final_output) => {
                        let final_message = final_output_to_assistant_message(
                            &final_output,
                            &self.final_output_spec,
                        );
                        if (is_task_solicitation_message(&final_message)
                            || is_meta_protocol_completion_message(&final_message))
                            && should_block_task_solicitation(latest_user_message.as_deref())
                        {
                            let latest_preview = latest_user_message
                                .as_deref()
                                .map(|text| text.replace('\n', " "))
                                .unwrap_or_default();
                            let latest_preview =
                                latest_preview.chars().take(220).collect::<String>();
                            let mut policy_error = format!(
                                "[Policy Error] Do not ask the user to provide a task when a latest user message exists. Latest user message: {latest_preview:?}. Respond directly to that message."
                            );
                            if self.settings.verbose && !stdout.trim().is_empty() {
                                policy_error = format!("{stdout}\n{policy_error}");
                            }
                            history.push(ReplEntry {
                                reasoning: action.reasoning,
                                code,
                                output: format_output(
                                    &policy_error,
                                    self.settings.max_output_chars,
                                ),
                            });
                            return IterationOutcome::Continue;
                        }
                        if likely_workspace_request && !code_uses_external_tools {
                            let latest_preview = latest_user_message
                                .as_deref()
                                .map(|text| text.replace('\n', " "))
                                .unwrap_or_default();
                            let latest_preview =
                                latest_preview.chars().take(220).collect::<String>();
                            let policy_error = format!(
                                "[Policy Error] Latest user request looks like a workspace/code action ({latest_preview:?}), but this step did not use Codex tools. Use tools to perform and verify the change before SUBMIT."
                            );
                            history.push(ReplEntry {
                                reasoning: action.reasoning,
                                code,
                                output: format_output(
                                    &policy_error,
                                    self.settings.max_output_chars,
                                ),
                            });
                            return IterationOutcome::Continue;
                        }
                        if contains_unresolved_tool_wrapper_metadata(&final_message) {
                            let policy_error = "[Policy Error] Final response still contains unresolved tool wrapper metadata (`Chunk ID`). Re-run and summarize concrete outcomes instead of returning wrapper text.".to_string();
                            history.push(ReplEntry {
                                reasoning: action.reasoning,
                                code,
                                output: format_output(
                                    &policy_error,
                                    self.settings.max_output_chars,
                                ),
                            });
                            return IterationOutcome::Continue;
                        }
                        if likely_workspace_request && workspace_execution_failed(&stdout) {
                            let policy_error = "[Policy Error] Workspace action appears to have failed. Re-run with concrete file paths and successful command verification before SUBMIT.".to_string();
                            history.push(ReplEntry {
                                reasoning: action.reasoning,
                                code,
                                output: format_output(
                                    &policy_error,
                                    self.settings.max_output_chars,
                                ),
                            });
                            return IterationOutcome::Continue;
                        }
                        let mut final_output_line = format!(
                            "FINAL: {}",
                            serde_json::to_string(&Value::Object(final_output))
                                .unwrap_or_else(|_| final_message.clone())
                        );
                        if self.settings.verbose && !stdout.trim().is_empty() {
                            final_output_line = format!("{stdout}\n{final_output_line}");
                        }
                        history.push(ReplEntry {
                            reasoning: action.reasoning,
                            code,
                            output: format_output(
                                &final_output_line,
                                self.settings.max_output_chars,
                            ),
                        });
                        IterationOutcome::Final(final_message)
                    }
                    Err(error) => {
                        history.push(ReplEntry {
                            reasoning: action.reasoning,
                            code,
                            output: format_output(&error, self.settings.max_output_chars),
                        });
                        IterationOutcome::Continue
                    }
                }
            }
        }
    }

    async fn emit_turn_diff_if_any(&self) {
        let unified_diff = {
            let mut tracker = self.tracker.lock().await;
            tracker.get_unified_diff()
        };
        if let Ok(Some(unified_diff)) = unified_diff {
            self.sess
                .send_event(
                    &self.turn_context,
                    EventMsg::TurnDiff(TurnDiffEvent { unified_diff }),
                )
                .await;
        }
    }

    async fn emit_final_assistant_message(&self, message: &str) {
        let response_item = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: message.to_string(),
            }],
            end_turn: None,
            phase: None,
        };
        self.sess
            .record_response_item_and_emit_turn_item(&self.turn_context, response_item)
            .await;
    }
}

#[derive(Debug)]
enum IterationOutcome {
    Continue,
    Final(String),
}

pub(crate) async fn maybe_run_turn(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: &[UserInput],
    client_session: &mut ModelClientSession,
    turn_metadata_header: Option<&str>,
    cancellation_token: CancellationToken,
) -> CodexResult<Option<String>> {
    let settings = NativeRlmSettings::from_config(turn_context.config.as_ref());
    if !settings.enabled() {
        return Ok(None);
    }

    let mcp_tools = sess
        .services
        .mcp_connection_manager
        .read()
        .await
        .list_all_tools()
        .or_cancel(&cancellation_token)
        .await?;

    let router = Arc::new(ToolRouter::from_config(
        &turn_context.tools_config,
        Some(
            mcp_tools
                .into_iter()
                .map(|(name, tool)| (name, tool.tool))
                .collect(),
        ),
        turn_context.dynamic_tools.as_slice(),
    ));

    let tracker: SharedTurnDiffTracker = Arc::new(Mutex::new(TurnDiffTracker::new()));
    let tool_runtime = ToolCallRuntime::new(
        Arc::clone(&router),
        Arc::clone(&sess),
        Arc::clone(&turn_context),
        Arc::clone(&tracker),
    );

    let tool_catalog = build_tool_catalog(router.as_ref(), sess.as_ref()).await;
    let base_instructions = sess.get_base_instructions().await;
    let rlm_base_instructions = compose_native_rlm_base_instructions(&base_instructions);
    let sub_llm_base_instructions = compose_sub_llm_base_instructions(&base_instructions);
    let final_output_spec =
        FinalOutputSpec::from_turn_schema(turn_context.final_output_json_schema.as_ref());
    if turn_context.final_output_json_schema.is_some() && !final_output_spec.schema_driven {
        sess.send_event(
            &turn_context,
            EventMsg::Warning(WarningEvent {
                message: "Native RLM could not infer schema output fields; falling back to SUBMIT(assistant_message=...)"
                    .to_string(),
            }),
        )
        .await;
    }

    let runner = NativeRlmRunner {
        sess,
        turn_context,
        client_session,
        turn_metadata_header,
        cancellation_token,
        settings,
        rlm_base_instructions,
        sub_llm_base_instructions,
        final_output_spec,
        tool_runtime,
        tool_catalog,
        llm_calls_used: 0,
        tracker,
    };

    runner.run(input).await
}

#[allow(clippy::too_many_arguments)]
async fn query_model_text_with_session(
    sess: &Session,
    turn_context: &TurnContext,
    client_session: &mut ModelClientSession,
    turn_metadata_header: Option<&str>,
    cancellation_token: &CancellationToken,
    user_prompt: &str,
    output_schema: Option<Value>,
    base_instructions: BaseInstructions,
    personality: Option<crate::config::types::Personality>,
) -> CodexResult<String> {
    let prompt = Prompt {
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: user_prompt.to_string(),
            }],
            end_turn: None,
            phase: None,
        }],
        tools: Vec::new(),
        parallel_tool_calls: false,
        base_instructions,
        personality,
        output_schema,
    };

    let mut stream = client_session
        .stream(
            &prompt,
            &turn_context.model_info,
            &turn_context.otel_manager,
            turn_context.reasoning_effort,
            turn_context.reasoning_summary,
            turn_metadata_header,
        )
        .or_cancel(cancellation_token)
        .await??;

    let mut completed_items: Vec<ResponseItem> = Vec::new();

    loop {
        let maybe_event = stream
            .next()
            .or_cancel(cancellation_token)
            .await
            .map_err(CodexErr::from)?;
        let event = match maybe_event {
            Some(event) => event?,
            None => {
                return Err(CodexErr::Stream(
                    "stream closed before response.completed".to_string(),
                    None,
                ));
            }
        };

        match event {
            ResponseEvent::OutputItemDone(item) => {
                completed_items.push(item);
            }
            ResponseEvent::ServerReasoningIncluded(included) => {
                sess.set_server_reasoning_included(included).await;
            }
            ResponseEvent::RateLimits(snapshot) => {
                sess.update_rate_limits(turn_context, snapshot).await;
            }
            ResponseEvent::Completed { token_usage, .. } => {
                sess.update_token_usage_info(turn_context, token_usage.as_ref())
                    .await;
                break;
            }
            _ => {}
        }
    }

    last_assistant_message(&completed_items).ok_or_else(|| {
        CodexErr::Fatal("native_rlm expected assistant text but received none".to_string())
    })
}

fn parse_bool(raw: Option<&str>) -> Option<bool> {
    let normalized = raw?.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn is_python_repl_timeout_error(error: &str) -> bool {
    error.starts_with(PYTHON_REPL_TIMEOUT_ERROR_PREFIX)
}

fn parse_positive_u32(raw: Option<&str>) -> Option<u32> {
    raw?.trim().parse::<u32>().ok().filter(|value| *value > 0)
}

fn parse_positive_usize(raw: Option<&str>) -> Option<usize> {
    raw?.trim().parse::<usize>().ok().filter(|value| *value > 0)
}

fn parse_positive_u64(raw: Option<&str>) -> Option<u64> {
    raw?.trim().parse::<u64>().ok().filter(|value| *value > 0)
}

fn compose_native_rlm_base_instructions(base_instructions: &BaseInstructions) -> BaseInstructions {
    if base_instructions
        .text
        .contains(DSPY_RLM_SYSTEM_APPENDIX_MARKER)
    {
        return base_instructions.clone();
    }

    let trimmed = base_instructions.text.trim_end();
    let text = if trimmed.is_empty() {
        DSPY_RLM_SYSTEM_APPENDIX.to_string()
    } else {
        format!("{trimmed}\n\n{DSPY_RLM_SYSTEM_APPENDIX}")
    };
    BaseInstructions { text }
}

fn compose_sub_llm_base_instructions(_base_instructions: &BaseInstructions) -> BaseInstructions {
    BaseInstructions {
        text: SUB_LLM_SYSTEM_INSTRUCTIONS.to_string(),
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn compose_conversation_history(
    history_for_prompt: &[ResponseItem],
    repl_history: &[ReplEntry],
) -> CodexResult<Vec<Value>> {
    let mut conversation_history = serde_json::to_value(history_for_prompt)?
        .as_array()
        .cloned()
        .unwrap_or_default();
    conversation_history.extend(repl_history.iter().enumerate().map(|(index, entry)| {
        json!({
            "type": "native_rlm_step",
            "step": index + 1,
            "reasoning": entry.reasoning,
            "code": entry.code,
            "output": entry.output,
        })
    }));
    Ok(conversation_history)
}

fn build_variables_map(
    history_for_prompt: &[ResponseItem],
    repl_history: &[ReplEntry],
) -> CodexResult<Map<String, Value>> {
    let conversation_history = compose_conversation_history(history_for_prompt, repl_history)?;

    let mut variables = Map::new();
    variables.insert(
        "conversation_history".to_string(),
        Value::Array(conversation_history),
    );
    Ok(variables)
}

fn content_text_from_value(content: &Value) -> String {
    match content {
        Value::String(text) => text.trim().to_string(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                let Value::Object(map) = item else {
                    return None;
                };
                ["text", "input_text", "output_text"]
                    .iter()
                    .find_map(|key| map.get(*key).and_then(Value::as_str))
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<String>>()
            .join(" ")
            .trim()
            .to_string(),
        _ => String::new(),
    }
}

fn latest_user_message_text(conversation_history: &Value) -> Option<String> {
    let Value::Array(entries) = conversation_history else {
        return None;
    };

    entries.iter().rev().find_map(|entry| {
        let Value::Object(map) = entry else {
            return None;
        };
        if map.get("role").and_then(Value::as_str) != Some("user") {
            return None;
        }
        map.get("content")
            .map(content_text_from_value)
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
    })
}

fn conversation_tail_summary(
    conversation_history: &Value,
    max_entries: usize,
    preview_chars: usize,
) -> String {
    let Value::Array(entries) = conversation_history else {
        return String::new();
    };

    let start = entries.len().saturating_sub(max_entries);
    entries
        .iter()
        .enumerate()
        .skip(start)
        .map(|(index, entry)| {
            let Value::Object(map) = entry else {
                return format!("[{index}] non-object entry");
            };
            let role = map.get("role").and_then(Value::as_str).unwrap_or("none");
            let text = map
                .get("content")
                .map(content_text_from_value)
                .unwrap_or_default();
            let preview = text
                .chars()
                .take(preview_chars)
                .collect::<String>()
                .replace('\n', " ");
            format!("[{index}] role={role} text={preview:?}")
        })
        .collect::<Vec<String>>()
        .join("\n")
}

fn iteration_one_code_has_required_history_inspection(code: &str) -> bool {
    let lowercase = code.to_ascii_lowercase();
    let references_history = lowercase.contains("conversation_history");
    let has_observable_inspection = lowercase.contains("print(")
        || lowercase.contains("pprint(")
        || lowercase.contains("llm_query(")
        || lowercase.contains("llm_query_batched(");
    references_history && has_observable_inspection
}

fn is_task_solicitation_message(message: &str) -> bool {
    let lowercase = message.to_ascii_lowercase();
    let direct_match = [
        "please provide a task",
        "provide the specific task",
        "provide a specific task",
        "share the concrete task",
        "please provide the concrete task",
        "please restate the specific task",
        "please restate the task",
        "restate the specific task",
        "specify the task",
        "share the specific task",
        "state the specific task",
        "what task",
        "proceed immediately",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle));
    if direct_match {
        return true;
    }

    let mentions_task = lowercase.contains("task");
    let asks_for_task = ["provide", "restate", "share", "specify", "state", "what"]
        .iter()
        .any(|needle| lowercase.contains(needle));
    (mentions_task && asks_for_task) || lowercase.contains("you want me to complete")
}

fn should_block_task_solicitation(latest_user_message: Option<&str>) -> bool {
    let Some(text) = latest_user_message
        .map(str::trim)
        .filter(|text| !text.is_empty())
    else {
        return false;
    };

    let lowercase = text.to_ascii_lowercase();
    !lowercase.starts_with("<environment_context>")
        && !lowercase.starts_with("<collaboration_mode>")
        && !lowercase.starts_with("# agents.md instructions")
        && !lowercase.starts_with("<instructions>")
}

fn is_likely_workspace_edit_request(text: &str) -> bool {
    let lowercase = text.to_ascii_lowercase();
    let has_action_verb = [
        "create",
        "make",
        "add",
        "edit",
        "update",
        "modify",
        "fix",
        "run",
        "test",
        "open",
        "implement",
        "improve",
        "feature",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle));
    let references_workspace = [
        "file",
        "folder",
        "directory",
        "repo",
        "code",
        "project",
        "app",
        "game",
        "website",
        "html",
        "javascript",
        "python",
        "rust",
        "it",
        "this",
        "that",
        "thing",
        "@",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle));
    has_action_verb && references_workspace
}

fn code_invokes_external_tool_alias(code: &str, aliases: &[String]) -> bool {
    aliases
        .iter()
        .filter(|alias| alias.as_str() != "llm_query" && alias.as_str() != "llm_query_batched")
        .map(|alias| format!("{alias}("))
        .any(|pattern| code.contains(&pattern))
}

fn is_meta_protocol_completion_message(message: &str) -> bool {
    let lowercase = message.to_ascii_lowercase();
    [
        "processed your rlm",
        "rlm-mode instruction",
        "protocol-compliant rlm execution",
        "inspected conversation history",
        "identified the latest user request",
        "identified your latest request",
        "completed the required submit step",
        "required submission field",
        "returned the required submission",
        "strict json output",
        "native rlm mode",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle))
}

fn workspace_execution_failed(stdout: &str) -> bool {
    let lowercase = stdout.to_ascii_lowercase();
    if lowercase.contains("no such file or directory")
        || lowercase.contains("can't open")
        || lowercase.contains("command not found")
    {
        return true;
    }

    stdout.lines().any(|line| {
        let trimmed = line.trim();
        if let Some(code_text) = trimmed.strip_prefix("Process exited with code ")
            && let Ok(code) = code_text.trim().parse::<i32>()
        {
            return code != 0;
        }
        false
    })
}

fn contains_unresolved_tool_wrapper_metadata(text: &str) -> bool {
    text.contains("Chunk ID:")
}

#[allow(clippy::too_many_arguments)]
fn render_action_prompt(
    turn_context: &TurnContext,
    variable_infos: &[String],
    history: &[ReplEntry],
    tools_documentation: &str,
    final_output_spec: &FinalOutputSpec,
    iteration: u32,
    max_iterations: u32,
    max_llm_calls: u32,
    latest_user_message: Option<&str>,
    conversation_tail: &str,
    python_command: &str,
) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "{ACTION_INSTRUCTIONS}");
    let _ = writeln!(
        prompt,
        "Expected final output fields:\n{}",
        final_output_spec.output_fields_description()
    );
    let _ = writeln!(
        prompt,
        "When done, call SUBMIT({}).",
        final_output_spec.submit_signature_hint()
    );
    let _ = writeln!(
        prompt,
        "Model: {}\nIteration: {}/{}\nSub-LLM budget: {max_llm_calls}",
        turn_context.model_info.slug,
        iteration + 1,
        max_iterations,
    );
    let _ = writeln!(
        prompt,
        "Interpreter hint: when invoking Python via `exec_command`, use `{python_command}` (not bare `python`)."
    );
    let latest_user = latest_user_message.unwrap_or("(none)");
    let _ = writeln!(prompt, "\nHost Conversation Anchors (authoritative):");
    let _ = writeln!(
        prompt,
        "Latest user message extracted from full `conversation_history`:\n---\n{latest_user}\n---"
    );
    if !conversation_tail.trim().is_empty() {
        let _ = writeln!(
            prompt,
            "Recent conversation tail (oldest -> newest):\n{conversation_tail}"
        );
    }
    let _ = writeln!(
        prompt,
        "Important: the controller prompt text is protocol guidance, not user conversation content."
    );
    let _ = writeln!(prompt, "\nVariables:");
    for info in variable_infos {
        let _ = writeln!(prompt, "- {info}");
    }

    if !tools_documentation.trim().is_empty() {
        let _ = writeln!(prompt, "\nCodex tool aliases available in Python:");
        let _ = writeln!(prompt, "{tools_documentation}");
    }

    let history_text = if history.is_empty() {
        "You have not executed code yet.".to_string()
    } else {
        history
            .iter()
            .enumerate()
            .map(|(index, entry)| entry.format(index, REPL_HISTORY_PROMPT_MAX_OUTPUT_CHARS))
            .collect::<Vec<String>>()
            .join("\n")
    };

    let _ = writeln!(prompt, "\nREPL history:\n{history_text}");
    prompt
}

fn render_extract_prompt(
    variable_infos: &[String],
    history: &[ReplEntry],
    final_output_spec: &FinalOutputSpec,
) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "{EXTRACT_INSTRUCTIONS}");
    let _ = writeln!(
        prompt,
        "Required final output fields:\n{}",
        final_output_spec.output_fields_description()
    );
    let _ = writeln!(prompt, "Return only a JSON object matching those fields.");

    let _ = writeln!(prompt, "\nVariables:");
    for info in variable_infos {
        let _ = writeln!(prompt, "- {info}");
    }

    let history_text = if history.is_empty() {
        "No REPL trajectory available.".to_string()
    } else {
        history
            .iter()
            .enumerate()
            .map(|(index, entry)| entry.format(index, REPL_HISTORY_PROMPT_MAX_OUTPUT_CHARS))
            .collect::<Vec<String>>()
            .join("\n")
    };
    let _ = writeln!(prompt, "\nREPL history:\n{history_text}");
    prompt
}

fn parse_action_step(raw: &str) -> CodexResult<ActionStep> {
    let parsed = serde_json::from_str::<ActionStep>(raw).or_else(|_| {
        extract_json_object(raw)
            .ok_or_else(|| {
                CodexErr::Fatal(format!(
                    "native_rlm action output was not valid JSON object: {raw}"
                ))
            })
            .and_then(|json_object| {
                serde_json::from_str::<ActionStep>(&json_object).map_err(CodexErr::from)
            })
    })?;
    Ok(parsed)
}

fn extract_json_object(raw: &str) -> Option<String> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end < start {
        return None;
    }
    Some(raw[start..=end].to_string())
}

fn strip_code_fences(code: &str) -> String {
    let trimmed = code.trim();
    if !trimmed.starts_with("```") || !trimmed.ends_with("```") {
        return trimmed.to_string();
    }

    let mut lines = trimmed.lines();
    let first = lines.next().unwrap_or_default();
    if !first.starts_with("```") {
        return trimmed.to_string();
    }

    let mut body_lines = lines.collect::<Vec<&str>>();
    if body_lines.last().is_some_and(|last| last.trim() == "```") {
        body_lines.pop();
    }
    body_lines.join("\n")
}

fn format_output(output: &str, max_chars: usize) -> String {
    if output.trim().is_empty() {
        return "(no output - did you forget to print?)".to_string();
    }

    let char_count = output.chars().count();
    if char_count <= max_chars {
        return output.to_string();
    }

    let truncated = output.chars().take(max_chars).collect::<String>();
    format!("{truncated}\n... (truncated)")
}

fn parse_final_output(
    output: Value,
    final_output_spec: &FinalOutputSpec,
) -> Result<Map<String, Value>, String> {
    let Value::Object(mut object) = output else {
        return Err(format!(
            "[Error] SUBMIT must receive a dict-like payload. Use SUBMIT({})",
            final_output_spec.submit_signature_hint()
        ));
    };

    let missing = final_output_spec
        .required_fields
        .iter()
        .filter(|name| !object.contains_key(*name))
        .cloned()
        .collect::<Vec<String>>();
    if !missing.is_empty() {
        return Err(format!(
            "[Error] Missing output fields: {}. Use SUBMIT({})",
            missing.join(", "),
            final_output_spec.submit_signature_hint()
        ));
    }

    if final_output_spec.enforce_no_additional_properties {
        let unknown = object
            .keys()
            .filter(|name| !final_output_spec.field_names.contains(*name))
            .cloned()
            .collect::<Vec<String>>();
        if !unknown.is_empty() {
            return Err(format!(
                "[Error] Unexpected output fields: {}. Allowed fields: {}",
                unknown.join(", "),
                final_output_spec.field_names.join(", ")
            ));
        }
    }

    let mut type_errors: Vec<String> = Vec::new();
    for (field_name, field_schema) in &final_output_spec.field_schemas {
        let Some(field_value) = object.get(field_name).cloned() else {
            continue;
        };
        if let Some(type_spec) = field_schema.get("type") {
            if let Some(coerced_value) = coerce_value_to_type_spec(&field_value, type_spec) {
                object.insert(field_name.clone(), coerced_value);
            } else {
                let expected =
                    schema_type_description(field_schema).unwrap_or_else(|| "unknown".to_string());
                type_errors.push(format!(
                    "{field_name}: expected {expected}, got {}",
                    json_type_name(&field_value)
                ));
            }
        }
    }
    if !type_errors.is_empty() {
        return Err(format!("[Type Error] {}", type_errors.join("; ")));
    }

    Ok(object)
}

fn final_output_to_assistant_message(
    output: &Map<String, Value>,
    final_output_spec: &FinalOutputSpec,
) -> String {
    if !final_output_spec.schema_driven
        && let Some(Value::String(message)) = output.get("assistant_message")
    {
        return message.clone();
    }

    serde_json::to_string(&Value::Object(output.clone())).unwrap_or_else(|_| {
        output
            .get("assistant_message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    })
}

fn schema_type_description(schema: &Value) -> Option<String> {
    let type_spec = schema.get("type")?;
    match type_spec {
        Value::String(value) => Some(value.clone()),
        Value::Array(values) => {
            let mut parsed = values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<String>>();
            if parsed.is_empty() {
                None
            } else {
                parsed.sort();
                parsed.dedup();
                Some(parsed.join(" | "))
            }
        }
        _ => None,
    }
}

fn coerce_value_to_type_spec(value: &Value, type_spec: &Value) -> Option<Value> {
    match type_spec {
        Value::String(type_name) => coerce_value_to_json_type(value, type_name),
        Value::Array(type_names) => type_names.iter().find_map(|entry| {
            entry
                .as_str()
                .and_then(|type_name| coerce_value_to_json_type(value, type_name))
        }),
        _ => Some(value.clone()),
    }
}

fn coerce_value_to_json_type(value: &Value, type_name: &str) -> Option<Value> {
    match type_name {
        "string" => match value {
            Value::String(_) => Some(value.clone()),
            _ => Some(Value::String(value_to_text(value))),
        },
        "number" => match value {
            Value::Number(_) => Some(value.clone()),
            Value::String(text) => text
                .trim()
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number),
            _ => None,
        },
        "integer" => match value {
            Value::Number(number) => {
                if number.as_i64().is_some() || number.as_u64().is_some() {
                    Some(value.clone())
                } else {
                    None
                }
            }
            Value::String(text) => text
                .trim()
                .parse::<i64>()
                .ok()
                .map(serde_json::Number::from)
                .map(Value::Number),
            _ => None,
        },
        "boolean" => match value {
            Value::Bool(_) => Some(value.clone()),
            Value::String(text) => {
                let normalized = text.trim().to_ascii_lowercase();
                match normalized.as_str() {
                    "true" => Some(Value::Bool(true)),
                    "false" => Some(Value::Bool(false)),
                    _ => None,
                }
            }
            _ => None,
        },
        "array" => match value {
            Value::Array(_) => Some(value.clone()),
            Value::String(text) => serde_json::from_str::<Value>(text.trim())
                .ok()
                .filter(Value::is_array),
            _ => None,
        },
        "object" => match value {
            Value::Object(_) => Some(value.clone()),
            Value::String(text) => serde_json::from_str::<Value>(text.trim())
                .ok()
                .filter(Value::is_object),
            _ => None,
        },
        "null" => value.is_null().then_some(Value::Null),
        _ => Some(value.clone()),
    }
}

async fn build_tool_catalog(router: &ToolRouter, session: &Session) -> ToolCatalog {
    let mut bindings_by_alias: HashMap<String, ToolBinding> = HashMap::new();
    let mut aliases: Vec<String> = vec!["llm_query".to_string(), "llm_query_batched".to_string()];
    let mut used_aliases: HashSet<String> = aliases.iter().cloned().collect();

    for spec in router.specs() {
        let Some((tool_name, description, kind)) = tool_info_from_spec(&spec, session).await else {
            continue;
        };

        let alias = make_unique_alias(&tool_name, &mut used_aliases);
        aliases.push(alias.clone());

        let signature = tool_signature(&alias, &spec);
        bindings_by_alias.insert(
            alias.clone(),
            ToolBinding {
                alias,
                tool_name,
                signature,
                description,
                kind,
            },
        );
    }

    let mut lines = Vec::new();
    lines.push("- llm_query(prompt: str) -> str".to_string());
    lines.push("- llm_query_batched(prompts: list[str]) -> list[str]".to_string());

    let mut sorted_bindings = bindings_by_alias
        .values()
        .cloned()
        .collect::<Vec<ToolBinding>>();
    sorted_bindings.sort_by(|left, right| left.alias.cmp(&right.alias));

    for binding in sorted_bindings {
        lines.push(format!(
            "- {} -> `{}`: {}",
            binding.signature, binding.tool_name, binding.description
        ));
    }

    ToolCatalog {
        bindings_by_alias,
        aliases,
        documentation: lines.join("\n"),
    }
}

async fn tool_info_from_spec(
    spec: &ToolSpec,
    session: &Session,
) -> Option<(String, String, ToolBindingKind)> {
    match spec {
        ToolSpec::Function(function) => {
            let name = function.name.clone();
            let description = function.description.clone();
            if let Some((server, tool)) = session.parse_mcp_tool_name(&name).await {
                return Some((name, description, ToolBindingKind::Mcp { server, tool }));
            }
            Some((name, description, ToolBindingKind::Function))
        }
        ToolSpec::Freeform(custom) => Some((
            custom.name.clone(),
            custom.description.clone(),
            ToolBindingKind::Custom,
        )),
        ToolSpec::LocalShell {} => Some((
            "local_shell".to_string(),
            "Runs a local shell command".to_string(),
            ToolBindingKind::LocalShell,
        )),
        ToolSpec::WebSearch { .. } => None,
    }
}

fn tool_signature(alias: &str, spec: &ToolSpec) -> String {
    match spec {
        ToolSpec::Function(function) => format_function_signature(alias, &function.parameters),
        ToolSpec::Freeform(_) => format!("{alias}(input: str)"),
        ToolSpec::LocalShell {} => format!(
            "{alias}(command: str | list[str], workdir: str | None = None, timeout_ms: int | None = None, sandbox_permissions: str | None = None, prefix_rule: list[str] | None = None, justification: str | None = None)"
        ),
        ToolSpec::WebSearch { .. } => format!("{alias}(query: str)"),
    }
}

fn format_function_signature(alias: &str, parameters: &JsonSchema) -> String {
    let JsonSchema::Object {
        properties,
        required,
        ..
    } = parameters
    else {
        return format!("{alias}(**kwargs)");
    };

    let required_names = required
        .as_ref()
        .map(|names| names.iter().map(String::as_str).collect::<HashSet<&str>>())
        .unwrap_or_default();

    let args = properties
        .iter()
        .map(|(name, schema)| {
            let type_name = schema_to_python_type(schema);
            if required_names.contains(name.as_str()) {
                format!("{name}: {type_name}")
            } else {
                format!("{name}: {type_name} | None = None")
            }
        })
        .collect::<Vec<String>>()
        .join(", ");
    format!("{alias}({args})")
}

fn schema_to_python_type(schema: &JsonSchema) -> String {
    match schema {
        JsonSchema::Boolean { .. } => "bool".to_string(),
        JsonSchema::String { .. } => "str".to_string(),
        JsonSchema::Number { .. } => "float".to_string(),
        JsonSchema::Array { items, .. } => format!("list[{}]", schema_to_python_type(items)),
        JsonSchema::Object { .. } => "dict[str, Any]".to_string(),
    }
}

fn make_unique_alias(tool_name: &str, used_aliases: &mut HashSet<String>) -> String {
    let base = sanitize_python_identifier(tool_name);
    if !used_aliases.contains(&base) {
        used_aliases.insert(base.clone());
        return base;
    }

    let mut suffix = 2u32;
    loop {
        let candidate = format!("{base}_{suffix}");
        if !used_aliases.contains(&candidate) {
            used_aliases.insert(candidate.clone());
            return candidate;
        }
        suffix += 1;
    }
}

fn sanitize_python_identifier(name: &str) -> String {
    let mut sanitized = String::new();
    for (index, character) in name.chars().enumerate() {
        let valid = character == '_' || character.is_ascii_alphanumeric();
        if valid {
            if index == 0 && character.is_ascii_digit() {
                sanitized.push('_');
            }
            sanitized.push(character);
        } else {
            sanitized.push('_');
        }
    }

    if sanitized.is_empty() {
        sanitized.push_str("tool");
    }

    if matches!(
        sanitized.as_str(),
        "llm_query"
            | "llm_query_batched"
            | "SUBMIT"
            | "print"
            | "class"
            | "def"
            | "for"
            | "while"
            | "if"
            | "else"
            | "try"
            | "except"
            | "return"
    ) {
        format!("{sanitized}_tool")
    } else {
        sanitized
    }
}

fn normalize_tool_call_arguments_for_native_rlm(
    binding: &ToolBinding,
    mut args: Vec<Value>,
    mut kwargs: Map<String, Value>,
    python_command: &str,
) -> (Vec<Value>, Map<String, Value>) {
    if binding.tool_name != "exec_command" {
        return (args, kwargs);
    }

    if let Some(cmd_value) = kwargs.get_mut("cmd")
        && let Some(cmd) = cmd_value.as_str()
        && let Some(rewritten_cmd) = rewrite_leading_python_command(cmd, python_command)
    {
        *cmd_value = Value::String(rewritten_cmd);
        return (args, kwargs);
    }

    if kwargs.is_empty()
        && let Some(Value::Object(object_args)) = args.first_mut()
        && let Some(cmd_value) = object_args.get_mut("cmd")
        && let Some(cmd) = cmd_value.as_str()
        && let Some(rewritten_cmd) = rewrite_leading_python_command(cmd, python_command)
    {
        *cmd_value = Value::String(rewritten_cmd);
    }

    (args, kwargs)
}

fn rewrite_leading_python_command(cmd: &str, python_command: &str) -> Option<String> {
    let leading_whitespace_length = cmd
        .chars()
        .take_while(|character| character.is_whitespace())
        .count();
    let (leading_whitespace, trimmed_command) = cmd.split_at(leading_whitespace_length);

    let first_token_end = trimmed_command
        .char_indices()
        .find_map(|(index, character)| character.is_whitespace().then_some(index))
        .unwrap_or(trimmed_command.len());
    let (first_token, remainder) = trimmed_command.split_at(first_token_end);

    if first_token != "python" {
        return None;
    }

    let normalized_python_command = python_command.trim();
    if normalized_python_command.is_empty() {
        return None;
    }

    Some(format!(
        "{leading_whitespace}{normalized_python_command}{remainder}"
    ))
}

fn build_tool_payload(
    binding: &ToolBinding,
    args: Vec<Value>,
    kwargs: Map<String, Value>,
) -> Result<ToolPayload, String> {
    match &binding.kind {
        ToolBindingKind::Function => {
            let arguments = arguments_from_call(args, kwargs)?;
            let arguments = serde_json::to_string(&arguments).map_err(|err| err.to_string())?;
            Ok(ToolPayload::Function { arguments })
        }
        ToolBindingKind::Mcp { server, tool } => {
            let arguments = arguments_from_call(args, kwargs)?;
            let raw_arguments = serde_json::to_string(&arguments).map_err(|err| err.to_string())?;
            Ok(ToolPayload::Mcp {
                server: server.clone(),
                tool: tool.clone(),
                raw_arguments,
            })
        }
        ToolBindingKind::Custom => {
            let input = if let Some(input) = kwargs.get("input") {
                value_to_text(input)
            } else if args.len() == 1 {
                value_to_text(&args[0])
            } else {
                return Err(
                    "custom tool calls must pass `input` or a single positional argument"
                        .to_string(),
                );
            };
            Ok(ToolPayload::Custom { input })
        }
        ToolBindingKind::LocalShell => {
            let mut call_kwargs = kwargs;
            normalize_local_shell_command(&mut call_kwargs)?;
            let value = Value::Object(call_kwargs);
            let params: ShellToolCallParams =
                serde_json::from_value(value).map_err(|err| err.to_string())?;
            Ok(ToolPayload::LocalShell { params })
        }
    }
}

fn normalize_local_shell_command(kwargs: &mut Map<String, Value>) -> Result<(), String> {
    let Some(command) = kwargs.get_mut("command") else {
        return Err("local_shell requires `command`".to_string());
    };

    if command.is_array() {
        return Ok(());
    }

    let Some(command_text) = command.as_str() else {
        return Err("local_shell.command must be a list of strings or a string".to_string());
    };

    #[cfg(target_os = "windows")]
    let command_parts = vec![
        Value::String("powershell.exe".to_string()),
        Value::String("-Command".to_string()),
        Value::String(command_text.to_string()),
    ];

    #[cfg(not(target_os = "windows"))]
    let command_parts = vec![
        Value::String("bash".to_string()),
        Value::String("-lc".to_string()),
        Value::String(command_text.to_string()),
    ];

    *command = Value::Array(command_parts);
    Ok(())
}

fn arguments_from_call(args: Vec<Value>, kwargs: Map<String, Value>) -> Result<Value, String> {
    if !kwargs.is_empty() {
        return Ok(Value::Object(kwargs));
    }

    if args.is_empty() {
        return Ok(Value::Object(Map::new()));
    }

    if args.len() == 1 && args[0].is_object() {
        return Ok(args[0].clone());
    }

    Err("tool calls must pass keyword args or a single object positional arg".to_string())
}

fn response_input_item_to_value(item: ResponseInputItem) -> Result<Value, String> {
    match item {
        ResponseInputItem::FunctionCallOutput { output, .. } => {
            Ok(Value::String(output.body.to_text().unwrap_or_default()))
        }
        ResponseInputItem::CustomToolCallOutput { output, .. } => Ok(Value::String(output)),
        ResponseInputItem::McpToolCallOutput { result, .. } => match result {
            Ok(value) => serde_json::to_value(value).map_err(|err| err.to_string()),
            Err(err) => Err(err),
        },
        ResponseInputItem::Message { content, .. } => Ok(Value::String(
            content
                .iter()
                .filter_map(|item| match item {
                    ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                        Some(text.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<&str>>()
                .join("\n"),
        )),
    }
}

fn native_rlm_tool_call_to_response_item(call: &ToolCall) -> ResponseItem {
    match &call.payload {
        ToolPayload::Custom { input } => ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id: call.call_id.clone(),
            name: call.tool_name.clone(),
            input: input.clone(),
        },
        ToolPayload::Function { arguments } => ResponseItem::FunctionCall {
            id: None,
            name: call.tool_name.clone(),
            arguments: arguments.clone(),
            call_id: call.call_id.clone(),
        },
        ToolPayload::Mcp { raw_arguments, .. } => ResponseItem::FunctionCall {
            id: None,
            name: call.tool_name.clone(),
            arguments: raw_arguments.clone(),
            call_id: call.call_id.clone(),
        },
        ToolPayload::LocalShell { params } => ResponseItem::FunctionCall {
            id: None,
            name: call.tool_name.clone(),
            arguments: local_shell_call_arguments_string(params),
            call_id: call.call_id.clone(),
        },
    }
}

fn local_shell_call_arguments_string(params: &ShellToolCallParams) -> String {
    let mut arguments = Map::new();
    arguments.insert(
        "command".to_string(),
        Value::Array(
            params
                .command
                .iter()
                .cloned()
                .map(Value::String)
                .collect::<Vec<Value>>(),
        ),
    );
    if let Some(workdir) = &params.workdir {
        arguments.insert("workdir".to_string(), Value::String(workdir.clone()));
    }
    if let Some(timeout_ms) = params.timeout_ms {
        arguments.insert("timeout_ms".to_string(), Value::Number(timeout_ms.into()));
    }
    if let Some(prefix_rule) = &params.prefix_rule {
        arguments.insert(
            "prefix_rule".to_string(),
            Value::Array(
                prefix_rule
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect::<Vec<Value>>(),
            ),
        );
    }
    if let Some(justification) = &params.justification {
        arguments.insert(
            "justification".to_string(),
            Value::String(justification.clone()),
        );
    }
    serde_json::to_string(&Value::Object(arguments)).unwrap_or_else(|_| "{}".to_string())
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    }
}

fn extract_prompt_argument(args: Vec<Value>, kwargs: Map<String, Value>) -> Result<String, String> {
    if let Some(prompt) = kwargs.get("prompt") {
        return prompt
            .as_str()
            .map(ToOwned::to_owned)
            .ok_or_else(|| "llm_query(prompt=...) expects prompt to be a string".to_string());
    }

    if args.len() == 1 {
        return args[0]
            .as_str()
            .map(ToOwned::to_owned)
            .ok_or_else(|| "llm_query expects first positional arg to be a string".to_string());
    }

    Err("llm_query requires exactly one prompt".to_string())
}

fn extract_prompts_argument(
    args: Vec<Value>,
    kwargs: Map<String, Value>,
) -> Result<Vec<String>, String> {
    let prompts_value = kwargs
        .get("prompts")
        .cloned()
        .or_else(|| args.first().cloned())
        .ok_or_else(|| "llm_query_batched requires prompts".to_string())?;

    let Value::Array(prompts) = prompts_value else {
        return Err("llm_query_batched prompts must be a list".to_string());
    };

    let mut parsed_prompts = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let Some(prompt_text) = prompt.as_str() else {
            return Err("llm_query_batched prompts must contain only strings".to_string());
        };
        parsed_prompts.push(prompt_text.to_string());
    }

    Ok(parsed_prompts)
}

fn last_assistant_message(items: &[ResponseItem]) -> Option<String> {
    items.iter().rev().find_map(|item| {
        let ResponseItem::Message { role, content, .. } = item else {
            return None;
        };
        if role != "assistant" {
            return None;
        }

        let text = content
            .iter()
            .filter_map(|entry| match entry {
                ContentItem::OutputText { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<&str>>()
            .join("\n");

        if text.is_empty() { None } else { Some(text) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_config;
    use pretty_assertions::assert_eq;

    #[test]
    fn native_rlm_settings_defaults() {
        let settings =
            NativeRlmSettings::from_raw_values(None, None, None, None, None, None, None, None);
        assert_eq!(
            settings,
            NativeRlmSettings {
                enabled: false,
                max_iterations: DEFAULT_MAX_ITERATIONS,
                max_llm_calls: DEFAULT_MAX_LLM_CALLS,
                llm_batch_concurrency: DEFAULT_LLM_BATCH_CONCURRENCY,
                max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
                exec_timeout_ms: DEFAULT_EXEC_TIMEOUT_MS,
                python_command: DEFAULT_PYTHON_COMMAND.to_string(),
                verbose: false,
            }
        );
    }

    #[test]
    fn native_rlm_settings_from_config_values() {
        let settings = NativeRlmSettings::from_config_values(Some(&NativeRlmToml {
            enabled: Some(true),
            max_iterations: Some(8),
            max_llm_calls: Some(9),
            llm_batch_concurrency: Some(3),
            max_output_chars: Some(1234),
            exec_timeout_ms: Some(4567),
            python_command: Some("python3 -X dev".to_string()),
            verbose: Some(true),
        }));
        assert_eq!(
            settings,
            NativeRlmSettings {
                enabled: true,
                max_iterations: 8,
                max_llm_calls: 9,
                llm_batch_concurrency: 3,
                max_output_chars: 1234,
                exec_timeout_ms: 4567,
                python_command: "python3 -X dev".to_string(),
                verbose: true,
            }
        );
    }

    #[test]
    fn native_rlm_settings_apply_env_overrides_changes_only_overridden_fields() {
        let mut settings = NativeRlmSettings::from_config_values(Some(&NativeRlmToml {
            enabled: Some(true),
            max_iterations: Some(8),
            max_llm_calls: Some(9),
            llm_batch_concurrency: Some(3),
            max_output_chars: Some(1234),
            exec_timeout_ms: Some(4567),
            python_command: Some("python3".to_string()),
            verbose: Some(false),
        }));

        settings.apply_env_overrides(NativeRlmEnvOverrides {
            enabled: Some(false),
            max_iterations: None,
            max_llm_calls: Some(99),
            llm_batch_concurrency: None,
            max_output_chars: None,
            exec_timeout_ms: Some(9000),
            python_command: Some("python3 -X dev".to_string()),
            verbose: Some(true),
        });

        assert_eq!(
            settings,
            NativeRlmSettings {
                enabled: false,
                max_iterations: 8,
                max_llm_calls: 99,
                llm_batch_concurrency: 3,
                max_output_chars: 1234,
                exec_timeout_ms: 9000,
                python_command: "python3 -X dev".to_string(),
                verbose: true,
            }
        );
    }

    #[test]
    fn should_disable_history_truncation_uses_config_settings() {
        let mut config = test_config();
        config.native_rlm = NativeRlmToml {
            enabled: Some(true),
            ..NativeRlmToml::default()
        };

        assert!(should_disable_history_truncation(&config));

        config.native_rlm.enabled = Some(false);
        assert!(!should_disable_history_truncation(&config));
    }

    #[test]
    fn native_rlm_settings_parsing() {
        let settings = NativeRlmSettings::from_raw_values(
            Some("true"),
            Some("8"),
            Some("9"),
            Some("3"),
            Some("1234"),
            Some("4567"),
            Some("python3 -X dev"),
            Some("1"),
        );
        assert_eq!(
            settings,
            NativeRlmSettings {
                enabled: true,
                max_iterations: 8,
                max_llm_calls: 9,
                llm_batch_concurrency: 3,
                max_output_chars: 1234,
                exec_timeout_ms: 4567,
                python_command: "python3 -X dev".to_string(),
                verbose: true,
            }
        );
    }

    #[test]
    fn native_rlm_status_message_is_multiline() {
        let settings = NativeRlmSettings::from_raw_values(
            Some("true"),
            Some("8"),
            Some("9"),
            Some("3"),
            Some("1234"),
            Some("4567"),
            Some("python3 -X dev"),
            Some("1"),
        );

        let status = settings
            .status_message()
            .expect("enabled settings should emit status");
        assert!(status.contains('\n'));
        assert!(status.contains("max_iterations=8"));
        assert!(status.contains("exec_timeout_ms=4567"));
    }

    #[test]
    fn compose_native_rlm_base_instructions_appends_dspy_guidance() {
        let base = BaseInstructions {
            text: "base instructions".to_string(),
        };

        let composed = compose_native_rlm_base_instructions(&base);
        assert!(composed.text.starts_with("base instructions"));
        assert!(composed.text.contains(DSPY_RLM_SYSTEM_APPENDIX_MARKER));
        assert!(
            composed
                .text
                .contains("preserving Codex assistant behavior")
        );
    }

    #[test]
    fn native_rlm_prompts_include_chronology_guardrails() {
        assert!(ACTION_INSTRUCTIONS.contains("latest unresolved request"));
        assert!(ACTION_INSTRUCTIONS.contains("tail-first"));
        assert!(ACTION_INSTRUCTIONS.contains("Iteration 1 is mandatory inspection-only"));
        assert!(ACTION_INSTRUCTIONS.contains("reverse scan from end"));
        assert!(ACTION_INSTRUCTIONS.contains("latest `role==\"user\"`"));
        assert!(ACTION_INSTRUCTIONS.contains("Do not ask for a new task"));
        assert!(ACTION_INSTRUCTIONS.contains("Non-coding requests are valid tasks"));
        assert!(ACTION_INSTRUCTIONS.contains("You can execute tools from this REPL"));
        assert!(ACTION_INSTRUCTIONS.contains("For chronology requests"));
        assert!(ACTION_INSTRUCTIONS.contains("always use `cmd=...`"));
        assert!(ACTION_INSTRUCTIONS.contains("kwargs only"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("first/last/earlier/previous-message"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("Core principles"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("latest unresolved user request"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("Iteration 1: inspect-only"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("tail context"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("continuation language"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("You can use tools in this REPL"));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("fallback \"please provide a task\""));
        assert!(DSPY_RLM_SYSTEM_APPENDIX.contains("kwargs-only tool calls"));
        assert!(SUB_LLM_SYSTEM_INSTRUCTIONS.contains("tail-first interpretation"));
    }

    #[test]
    fn compose_native_rlm_base_instructions_is_idempotent() {
        let base = BaseInstructions {
            text: format!("base\n\n{DSPY_RLM_SYSTEM_APPENDIX}"),
        };

        let composed = compose_native_rlm_base_instructions(&base);
        assert_eq!(composed, base);
    }

    #[test]
    fn compose_sub_llm_base_instructions_uses_plain_sub_model_prompt() {
        let base = BaseInstructions {
            text: "ignored".to_string(),
        };

        let composed = compose_sub_llm_base_instructions(&base);
        assert_eq!(composed.text, SUB_LLM_SYSTEM_INSTRUCTIONS);
        assert!(!composed.text.contains(DSPY_RLM_SYSTEM_APPENDIX_MARKER));
    }

    #[test]
    fn python_runner_writes_real_newline_delimiters() {
        assert!(
            PYTHON_REPL_RUNNER
                .contains("_HOST_STDOUT.write(json.dumps(obj, ensure_ascii=False) + \"\\n\")",)
        );
        assert!(
            !PYTHON_REPL_RUNNER
                .contains("_HOST_STDOUT.write(json.dumps(obj, ensure_ascii=False) + \"\\\\n\")",)
        );
    }

    #[test]
    fn python_runner_routes_protocol_json_to_host_stdout() {
        assert!(PYTHON_REPL_RUNNER.contains("_HOST_STDOUT = sys.stdout"));
        assert!(
            PYTHON_REPL_RUNNER.contains("_HOST_STDOUT.write(json.dumps(obj, ensure_ascii=False)")
        );
    }

    #[test]
    fn detects_python_repl_timeout_errors() {
        assert!(is_python_repl_timeout_error(
            "native_rlm Python REPL timed out after 180000 ms while waiting for output"
        ));
        assert!(!is_python_repl_timeout_error(
            "RuntimeError: invalid tool response from host"
        ));
    }

    #[test]
    fn sanitize_alias_keeps_identifiers_and_rewrites_invalid() {
        assert_eq!(sanitize_python_identifier("shell_command"), "shell_command");
        assert_eq!(
            sanitize_python_identifier("tool.with.dots"),
            "tool_with_dots"
        );
        assert_eq!(sanitize_python_identifier("123abc"), "_123abc");
        assert_eq!(sanitize_python_identifier("llm_query"), "llm_query_tool");
    }

    #[test]
    fn make_unique_alias_appends_suffixes() {
        let mut used = HashSet::from(["shell".to_string()]);
        let alias = make_unique_alias("shell", &mut used);
        assert_eq!(alias, "shell_2");
        assert!(used.contains("shell_2"));
    }

    #[test]
    fn format_function_signature_uses_schema_types_and_optionality() {
        let parameters = JsonSchema::Object {
            properties: std::collections::BTreeMap::from([
                (
                    "limit".to_string(),
                    JsonSchema::Number { description: None },
                ),
                (
                    "query".to_string(),
                    JsonSchema::String { description: None },
                ),
                (
                    "tags".to_string(),
                    JsonSchema::Array {
                        items: Box::new(JsonSchema::String { description: None }),
                        description: None,
                    },
                ),
            ]),
            required: Some(vec!["query".to_string()]),
            additional_properties: None,
        };

        let signature = format_function_signature("search", &parameters);
        assert_eq!(
            signature,
            "search(limit: float | None = None, query: str, tags: list[str] | None = None)"
        );
    }

    #[test]
    fn format_function_signature_falls_back_for_non_object_schema() {
        let signature =
            format_function_signature("tool", &JsonSchema::String { description: None });
        assert_eq!(signature, "tool(**kwargs)");
    }

    #[test]
    fn strip_code_fences_supports_markdown_blocks() {
        let code = "```python\nprint('x')\n```";
        assert_eq!(strip_code_fences(code), "print('x')");

        let plain = "print('x')";
        assert_eq!(strip_code_fences(plain), "print('x')");
    }

    #[test]
    fn parse_final_output_requires_assistant_message_string() {
        let spec = FinalOutputSpec::default_assistant_message();
        let ok = parse_final_output(json!({ "assistant_message": "done" }), &spec);
        assert_eq!(
            ok,
            Ok(Map::from_iter([(
                "assistant_message".to_string(),
                Value::String("done".to_string()),
            )]))
        );

        let missing = parse_final_output(json!({ "answer": "nope" }), &spec);
        assert_eq!(
            missing,
            Err(
                "[Error] Missing output fields: assistant_message. Use SUBMIT(assistant_message=...)"
                    .to_string(),
            )
        );
    }

    #[test]
    fn parse_final_output_enforces_schema_types() {
        let spec = FinalOutputSpec::from_turn_schema(Some(&json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" },
                "score": { "type": "integer" }
            },
            "required": ["summary", "score"],
            "additionalProperties": false
        })));

        let parsed = parse_final_output(
            json!({
                "summary": "ok",
                "score": 5
            }),
            &spec,
        );
        assert_eq!(
            parsed,
            Ok(Map::from_iter([
                ("summary".to_string(), Value::String("ok".to_string())),
                ("score".to_string(), Value::Number(5.into())),
            ]))
        );

        let coerced = parse_final_output(
            json!({
                "summary": "ok",
                "score": "7"
            }),
            &spec,
        );
        assert_eq!(
            coerced,
            Ok(Map::from_iter([
                ("summary".to_string(), Value::String("ok".to_string())),
                ("score".to_string(), Value::Number(7.into())),
            ]))
        );

        let type_error = parse_final_output(
            json!({
                "summary": "ok",
                "score": "not-int"
            }),
            &spec,
        );
        assert_eq!(
            type_error,
            Err("[Type Error] score: expected integer, got string".to_string())
        );
    }

    #[test]
    fn extract_json_object_from_wrapped_text() {
        let raw = "noise {\"reasoning\":\"r\",\"code\":\"c\"} tail";
        let extracted = extract_json_object(raw);
        assert_eq!(
            extracted,
            Some("{\"reasoning\":\"r\",\"code\":\"c\"}".to_string())
        );
    }

    #[test]
    fn parse_action_step_accepts_json_object() {
        let raw = "{\"reasoning\":\"inspect\",\"code\":\"print(1)\"}";
        let parsed = parse_action_step(raw).expect("action should parse");
        assert_eq!(
            parsed,
            ActionStep {
                reasoning: "inspect".to_string(),
                code: "print(1)".to_string(),
            }
        );
    }

    #[test]
    fn format_output_truncates_long_text() {
        let output = "abcdef";
        let formatted = format_output(output, 3);
        assert_eq!(formatted, "abc\n... (truncated)");
    }

    #[test]
    fn repl_entry_format_truncates_for_prompt_rendering() {
        let entry = ReplEntry {
            reasoning: "inspect".to_string(),
            code: "print(1)".to_string(),
            output: "abcdef".to_string(),
        };
        let formatted = entry.format(0, 3);
        assert!(formatted.contains("=== Step 1 ==="));
        assert!(formatted.contains("Output (6 chars):"));
        assert!(formatted.contains("abc"));
        assert!(formatted.contains("... (truncated to 3/6)"));
    }

    #[test]
    fn compose_conversation_history_appends_repl_steps() {
        let prompt_history = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "hello".to_string(),
            }],
            end_turn: None,
            phase: None,
        }];
        let repl_history = vec![ReplEntry {
            reasoning: "inspect".to_string(),
            code: "print(1)".to_string(),
            output: "1".to_string(),
        }];

        let history = compose_conversation_history(&prompt_history, &repl_history)
            .expect("compose conversation history");
        assert_eq!(history.len(), 2);
        assert_eq!(
            history[1],
            json!({
                "type": "native_rlm_step",
                "step": 1,
                "reasoning": "inspect",
                "code": "print(1)",
                "output": "1",
            })
        );
    }

    #[test]
    fn native_rlm_tool_call_to_response_item_maps_function_style_payloads() {
        let function_call = ToolCall {
            tool_name: "exec_command".to_string(),
            call_id: "call-function".to_string(),
            payload: ToolPayload::Function {
                arguments: "{\"cmd\":\"ls\"}".to_string(),
            },
        };
        assert_eq!(
            native_rlm_tool_call_to_response_item(&function_call),
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                arguments: "{\"cmd\":\"ls\"}".to_string(),
                call_id: "call-function".to_string(),
            }
        );

        let mcp_call = ToolCall {
            tool_name: "mcp__search".to_string(),
            call_id: "call-mcp".to_string(),
            payload: ToolPayload::Mcp {
                server: "docs".to_string(),
                tool: "search".to_string(),
                raw_arguments: "{\"q\":\"rust\"}".to_string(),
            },
        };
        assert_eq!(
            native_rlm_tool_call_to_response_item(&mcp_call),
            ResponseItem::FunctionCall {
                id: None,
                name: "mcp__search".to_string(),
                arguments: "{\"q\":\"rust\"}".to_string(),
                call_id: "call-mcp".to_string(),
            }
        );

        let local_shell_call = ToolCall {
            tool_name: "local_shell".to_string(),
            call_id: "call-shell".to_string(),
            payload: ToolPayload::LocalShell {
                params: ShellToolCallParams {
                    command: vec!["bash".to_string(), "-lc".to_string(), "pwd".to_string()],
                    workdir: Some("/tmp".to_string()),
                    timeout_ms: Some(42),
                    sandbox_permissions: None,
                    prefix_rule: None,
                    justification: None,
                },
            },
        };
        let local_shell_item = native_rlm_tool_call_to_response_item(&local_shell_call);
        let ResponseItem::FunctionCall {
            id,
            name,
            arguments,
            call_id,
        } = local_shell_item
        else {
            panic!("expected function call item for local shell payload");
        };
        assert_eq!(id, None);
        assert_eq!(name, "local_shell");
        assert_eq!(call_id, "call-shell");
        assert_eq!(
            serde_json::from_str::<Value>(&arguments)
                .expect("local shell arguments should be JSON"),
            json!({
                "command": ["bash", "-lc", "pwd"],
                "workdir": "/tmp",
                "timeout_ms": 42
            })
        );
    }

    #[test]
    fn native_rlm_tool_call_to_response_item_maps_custom_payload() {
        let call = ToolCall {
            tool_name: "apply_patch".to_string(),
            call_id: "call-custom".to_string(),
            payload: ToolPayload::Custom {
                input: "*** Begin Patch".to_string(),
            },
        };
        assert_eq!(
            native_rlm_tool_call_to_response_item(&call),
            ResponseItem::CustomToolCall {
                id: None,
                status: None,
                call_id: "call-custom".to_string(),
                name: "apply_patch".to_string(),
                input: "*** Begin Patch".to_string(),
            }
        );
    }

    #[test]
    fn build_variables_map_only_exposes_conversation_history() {
        let prompt_history = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "hello".to_string(),
            }],
            end_turn: None,
            phase: None,
        }];
        let repl_history = vec![ReplEntry {
            reasoning: "inspect".to_string(),
            code: "print(1)".to_string(),
            output: "1".to_string(),
        }];

        let variables =
            build_variables_map(&prompt_history, &repl_history).expect("build variable map");
        assert_eq!(variables.len(), 1);
        assert!(variables.contains_key("conversation_history"));
    }

    #[test]
    fn latest_user_message_text_prefers_newest_user_entry() {
        let history = json!([
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "older request" }]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "done" }]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "latest request" }]
            }
        ]);
        assert_eq!(
            latest_user_message_text(&history),
            Some("latest request".to_string())
        );
    }

    #[test]
    fn iteration_one_code_inspection_detector_requires_history_and_observable_signal() {
        assert!(iteration_one_code_has_required_history_inspection(
            "print(len(conversation_history))"
        ));
        assert!(iteration_one_code_has_required_history_inspection(
            "llm_query(str(conversation_history))"
        ));
        assert!(!iteration_one_code_has_required_history_inspection(
            "SUBMIT(assistant_message='please provide a task')"
        ));
        assert!(!iteration_one_code_has_required_history_inspection(
            "latest = conversation_history[-1]"
        ));
    }

    #[test]
    fn task_solicitation_guard_blocks_when_latest_user_message_exists() {
        assert!(is_task_solicitation_message("please provide a task"));
        assert!(is_task_solicitation_message(
            "Please provide a specific task you want me to complete."
        ));
        assert!(is_task_solicitation_message(
            "Please restate the specific task in one line if you want me to proceed immediately."
        ));
        assert!(should_block_task_solicitation(Some(
            "can you add some new features to the mario thing"
        )));
        assert!(!should_block_task_solicitation(Some(
            "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>"
        )));
        assert!(is_meta_protocol_completion_message(
            "Processed your RLM-mode instruction: inspected conversation history and completed the required submit step."
        ));
        assert!(is_meta_protocol_completion_message(
            "Completed: I identified your latest request as protocol-compliant RLM execution and returned the required submission field."
        ));
    }

    #[test]
    fn workspace_request_detector_and_tool_invocation_detector_work() {
        assert!(is_likely_workspace_edit_request("can you add world to it"));
        assert!(is_likely_workspace_edit_request(
            "create file hello.py with hello world"
        ));
        assert!(is_likely_workspace_edit_request(
            "can you add some new features to the mario thing"
        ));
        assert!(!is_likely_workspace_edit_request("hi"));

        let aliases = vec![
            "llm_query".to_string(),
            "llm_query_batched".to_string(),
            "exec_command".to_string(),
            "read_file".to_string(),
        ];
        assert!(code_invokes_external_tool_alias(
            "result = exec_command(cmd='ls')",
            &aliases
        ));
        assert!(!code_invokes_external_tool_alias(
            "result = llm_query('hi')",
            &aliases
        ));
        assert!(!code_invokes_external_tool_alias(
            "print('hello')",
            &aliases
        ));
        assert!(workspace_execution_failed(
            "Process exited with code 2\nOutput:\nrg: Chunk: No such file or directory"
        ));
        assert!(!workspace_execution_failed(
            "Process exited with code 0\nOutput:\nhello world"
        ));
        assert!(contains_unresolved_tool_wrapper_metadata(
            "Chunk ID: abc123\nWall time: 0.05 seconds"
        ));
        assert!(!contains_unresolved_tool_wrapper_metadata(
            "Updated index.html successfully."
        ));
    }

    #[test]
    fn rewrite_leading_python_command_rewrites_bare_python_only() {
        assert_eq!(
            rewrite_leading_python_command("python - <<'PY'\nprint('hi')\nPY", "python3"),
            Some("python3 - <<'PY'\nprint('hi')\nPY".to_string())
        );
        assert_eq!(
            rewrite_leading_python_command("  python script.py", "python3 -X dev"),
            Some("  python3 -X dev script.py".to_string())
        );
        assert_eq!(
            rewrite_leading_python_command("python3 script.py", "python3"),
            None
        );
    }

    #[test]
    fn normalize_tool_call_arguments_rewrites_exec_command_cmd_kwarg() {
        let binding = ToolBinding {
            alias: "exec_command".to_string(),
            tool_name: "exec_command".to_string(),
            signature: "exec_command(cmd: str)".to_string(),
            description: "Runs a command".to_string(),
            kind: ToolBindingKind::Function,
        };

        let (args, kwargs) = normalize_tool_call_arguments_for_native_rlm(
            &binding,
            Vec::new(),
            Map::from_iter([(
                "cmd".to_string(),
                Value::String("python - <<'PY'\nprint('ok')\nPY".to_string()),
            )]),
            "python3",
        );

        assert!(args.is_empty());
        assert_eq!(
            kwargs.get("cmd"),
            Some(&Value::String(
                "python3 - <<'PY'\nprint('ok')\nPY".to_string()
            ))
        );
    }

    #[test]
    fn normalize_tool_call_arguments_rewrites_exec_command_object_arg() {
        let binding = ToolBinding {
            alias: "exec_command".to_string(),
            tool_name: "exec_command".to_string(),
            signature: "exec_command(cmd: str)".to_string(),
            description: "Runs a command".to_string(),
            kind: ToolBindingKind::Function,
        };

        let (args, kwargs) = normalize_tool_call_arguments_for_native_rlm(
            &binding,
            vec![json!({
                "cmd": "python script.py",
                "workdir": "/tmp"
            })],
            Map::new(),
            "python3",
        );

        assert!(kwargs.is_empty());
        assert_eq!(
            args,
            vec![json!({
                "cmd": "python3 script.py",
                "workdir": "/tmp"
            })]
        );
    }
}
