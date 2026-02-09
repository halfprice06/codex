#![cfg(not(target_os = "windows"))]

use std::ffi::OsString;

use anyhow::Result;
use core_test_support::responses;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::sse;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use serial_test::serial;

const NATIVE_RLM_ENABLED_ENV: &str = "CODEX_NATIVE_RLM";
const NATIVE_RLM_MAX_ITERATIONS_ENV: &str = "CODEX_NATIVE_RLM_MAX_ITERATIONS";
const NATIVE_RLM_MAX_LLM_CALLS_ENV: &str = "CODEX_NATIVE_RLM_MAX_LLM_CALLS";

fn action_response_sse(response_id: &str, message_id: &str, reasoning: &str, code: &str) -> String {
    let payload = json!({
        "reasoning": reasoning,
        "code": code,
    })
    .to_string();
    sse(vec![
        ev_response_created(response_id),
        ev_assistant_message(message_id, &payload),
        ev_completed(response_id),
    ])
}

fn extract_total_length(prompt_text: &str) -> usize {
    prompt_text
        .lines()
        .find_map(|line| {
            line.strip_prefix("Total length: ")
                .and_then(|value| value.strip_suffix(" chars"))
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or_else(|| panic!("missing variable total-length line in prompt:\n{prompt_text}"))
}

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        // SAFETY: tests in this module run under the same serial key, so process env mutation is safe.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: guard restores env after each serially-executed test.
        unsafe {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[serial(native_rlm_env)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_rlm_passes_only_conversation_history_and_grows_it() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let _enabled_guard = EnvVarGuard::set(NATIVE_RLM_ENABLED_ENV, "1");
    let _iterations_guard = EnvVarGuard::set(NATIVE_RLM_MAX_ITERATIONS_ENV, "4");
    let _llm_calls_guard = EnvVarGuard::set(NATIVE_RLM_MAX_LLM_CALLS_ENV, "8");

    let server = responses::start_mock_server().await;
    let mut builder = test_codex().with_model("gpt-5");
    let test = builder.build(&server).await?;

    let first_mock = responses::mount_sse_once(
        &server,
        action_response_sse("resp-1", "msg-1", "inspect", "print('alpha')"),
    )
    .await;
    let second_mock = responses::mount_sse_once(
        &server,
        action_response_sse(
            "resp-2",
            "msg-2",
            "submit",
            "SUBMIT(assistant_message='history ok')",
        ),
    )
    .await;

    test.submit_turn("native rlm history variable check")
        .await?;

    let first_prompt = first_mock
        .single_request()
        .message_input_texts("user")
        .join("\n");
    let first_history_length = extract_total_length(&first_prompt);
    assert_eq!(first_prompt.matches("Variable: `").count(), 1);
    assert!(first_prompt.contains("Variable: `conversation_history`"));
    assert!(!first_prompt.contains("Variable: `latest_user_request`"));
    assert!(!first_prompt.contains("Variable: `cwd`"));
    assert!(!first_prompt.contains("Variable: `thread_id`"));
    assert!(first_prompt.contains("You have not executed code yet."));

    let second_prompt = second_mock
        .single_request()
        .message_input_texts("user")
        .join("\n");
    let second_history_length = extract_total_length(&second_prompt);
    assert_eq!(second_prompt.matches("Variable: `").count(), 1);
    assert!(second_prompt.contains("Variable: `conversation_history`"));
    assert!(second_history_length > first_history_length);
    assert!(second_prompt.contains("=== Step 1 ==="));
    assert!(second_prompt.contains("print('alpha')"));

    Ok(())
}
