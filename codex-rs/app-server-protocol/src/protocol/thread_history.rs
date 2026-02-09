use crate::protocol::v2::CollabAgentState;
use crate::protocol::v2::CollabAgentTool;
use crate::protocol::v2::CollabAgentToolCallStatus;
use crate::protocol::v2::CommandAction;
use crate::protocol::v2::CommandExecutionStatus;
use crate::protocol::v2::FileUpdateChange;
use crate::protocol::v2::McpToolCallError;
use crate::protocol::v2::McpToolCallResult;
use crate::protocol::v2::McpToolCallStatus;
use crate::protocol::v2::PatchApplyStatus;
use crate::protocol::v2::PatchChangeKind;
use crate::protocol::v2::ThreadItem;
use crate::protocol::v2::Turn;
use crate::protocol::v2::TurnError as V2TurnError;
use crate::protocol::v2::TurnError;
use crate::protocol::v2::TurnStatus;
use crate::protocol::v2::UserInput;
use crate::protocol::v2::WebSearchAction;
use codex_protocol::protocol::AgentReasoningEvent;
use codex_protocol::protocol::AgentReasoningRawContentEvent;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::ContextCompactedEvent;
use codex_protocol::protocol::ErrorEvent;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ExecCommandEndEvent;
use codex_protocol::protocol::ItemCompletedEvent;
use codex_protocol::protocol::McpToolCallEndEvent;
use codex_protocol::protocol::PatchApplyEndEvent;
use codex_protocol::protocol::ReviewOutputEvent;
use codex_protocol::protocol::ThreadRolledBackEvent;
use codex_protocol::protocol::TurnAbortedEvent;
use codex_protocol::protocol::UserMessageEvent;
use codex_protocol::protocol::ViewImageToolCallEvent;
use codex_protocol::protocol::WebSearchEndEvent;
use std::collections::HashMap;

/// Convert persisted [`EventMsg`] entries into a sequence of [`Turn`] values.
///
/// The purpose of this is to convert the EventMsgs persisted in a rollout file
/// into a sequence of Turns and ThreadItems, which allows the client to render
/// the historical messages when resuming a thread.
pub fn build_turns_from_event_msgs(events: &[EventMsg]) -> Vec<Turn> {
    let mut builder = ThreadHistoryBuilder::new();
    for event in events {
        builder.handle_event(event);
    }
    builder.finish()
}

struct ThreadHistoryBuilder {
    turns: Vec<Turn>,
    current_turn: Option<PendingTurn>,
    next_turn_index: i64,
    next_item_index: i64,
}

impl ThreadHistoryBuilder {
    fn new() -> Self {
        Self {
            turns: Vec::new(),
            current_turn: None,
            next_turn_index: 1,
            next_item_index: 1,
        }
    }

    fn finish(mut self) -> Vec<Turn> {
        self.finish_current_turn();
        self.turns
    }

    /// This function should handle all EventMsg variants that can be persisted in a rollout file.
    /// See `should_persist_event_msg` in `codex-rs/core/rollout/policy.rs`.
    fn handle_event(&mut self, event: &EventMsg) {
        match event {
            EventMsg::UserMessage(payload) => self.handle_user_message(payload),
            EventMsg::AgentMessage(payload) => self.handle_agent_message(payload.message.clone()),
            EventMsg::AgentReasoning(payload) => self.handle_agent_reasoning(payload),
            EventMsg::AgentReasoningRawContent(payload) => {
                self.handle_agent_reasoning_raw_content(payload)
            }
            EventMsg::WebSearchEnd(payload) => self.handle_web_search_end(payload),
            EventMsg::ExecCommandEnd(payload) => self.handle_exec_command_end(payload),
            EventMsg::PatchApplyEnd(payload) => self.handle_patch_apply_end(payload),
            EventMsg::McpToolCallEnd(payload) => self.handle_mcp_tool_call_end(payload),
            EventMsg::ViewImageToolCall(payload) => self.handle_view_image_tool_call(payload),
            EventMsg::CollabAgentSpawnEnd(payload) => self.handle_collab_agent_spawn_end(payload),
            EventMsg::CollabAgentInteractionEnd(payload) => {
                self.handle_collab_agent_interaction_end(payload)
            }
            EventMsg::CollabWaitingEnd(payload) => self.handle_collab_waiting_end(payload),
            EventMsg::CollabCloseEnd(payload) => self.handle_collab_close_end(payload),
            EventMsg::ContextCompacted(payload) => self.handle_context_compacted(payload),
            EventMsg::EnteredReviewMode(payload) => self.handle_entered_review_mode(payload),
            EventMsg::ExitedReviewMode(payload) => self.handle_exited_review_mode(payload),
            EventMsg::ItemCompleted(payload) => self.handle_item_completed(payload),
            EventMsg::Error(payload) => self.handle_error(payload),
            EventMsg::TokenCount(_) => {}
            EventMsg::ThreadRolledBack(payload) => self.handle_thread_rollback(payload),
            EventMsg::UndoCompleted(_) => {}
            EventMsg::TurnAborted(payload) => self.handle_turn_aborted(payload),
            _ => {}
        }
    }

    fn handle_user_message(&mut self, payload: &UserMessageEvent) {
        self.finish_current_turn();
        let mut turn = self.new_turn();
        let id = self.next_item_id();
        let content = self.build_user_inputs(payload);
        turn.items.push(ThreadItem::UserMessage { id, content });
        self.current_turn = Some(turn);
    }

    fn handle_agent_message(&mut self, text: String) {
        if text.is_empty() {
            return;
        }

        let id = self.next_item_id();
        self.ensure_turn()
            .items
            .push(ThreadItem::AgentMessage { id, text });
    }

    fn handle_agent_reasoning(&mut self, payload: &AgentReasoningEvent) {
        if payload.text.is_empty() {
            return;
        }

        // If the last item is a reasoning item, add the new text to the summary.
        if let Some(ThreadItem::Reasoning { summary, .. }) = self.ensure_turn().items.last_mut() {
            summary.push(payload.text.clone());
            return;
        }

        // Otherwise, create a new reasoning item.
        let id = self.next_item_id();
        self.ensure_turn().items.push(ThreadItem::Reasoning {
            id,
            summary: vec![payload.text.clone()],
            content: Vec::new(),
        });
    }

    fn handle_agent_reasoning_raw_content(&mut self, payload: &AgentReasoningRawContentEvent) {
        if payload.text.is_empty() {
            return;
        }

        // If the last item is a reasoning item, add the new text to the content.
        if let Some(ThreadItem::Reasoning { content, .. }) = self.ensure_turn().items.last_mut() {
            content.push(payload.text.clone());
            return;
        }

        // Otherwise, create a new reasoning item.
        let id = self.next_item_id();
        self.ensure_turn().items.push(ThreadItem::Reasoning {
            id,
            summary: Vec::new(),
            content: vec![payload.text.clone()],
        });
    }

    fn handle_item_completed(&mut self, payload: &ItemCompletedEvent) {
        if let codex_protocol::items::TurnItem::Plan(plan) = &payload.item
            && plan.text.is_empty()
        {
            return;
        }

        let item = ThreadItem::from(payload.item.clone());
        self.ensure_turn().items.push(item);
    }

    fn handle_web_search_end(&mut self, payload: &WebSearchEndEvent) {
        let item = ThreadItem::WebSearch {
            id: payload.call_id.clone(),
            query: payload.query.clone(),
            action: Some(WebSearchAction::from(payload.action.clone())),
        };
        self.ensure_turn().items.push(item);
    }

    fn handle_exec_command_end(&mut self, payload: &ExecCommandEndEvent) {
        let status = if payload.exit_code == 0 {
            CommandExecutionStatus::Completed
        } else {
            CommandExecutionStatus::Failed
        };
        let duration_ms = i64::try_from(payload.duration.as_millis()).unwrap_or(i64::MAX);
        let aggregated_output = if payload.aggregated_output.is_empty() {
            None
        } else {
            Some(payload.aggregated_output.clone())
        };
        let command = shlex::try_join(payload.command.iter().map(String::as_str))
            .unwrap_or_else(|_| payload.command.join(" "));
        let command_actions = payload
            .parsed_cmd
            .iter()
            .cloned()
            .map(CommandAction::from)
            .collect();
        let item = ThreadItem::CommandExecution {
            id: payload.call_id.clone(),
            command,
            cwd: payload.cwd.clone(),
            process_id: payload.process_id.clone(),
            status,
            command_actions,
            aggregated_output,
            exit_code: Some(payload.exit_code),
            duration_ms: Some(duration_ms),
        };
        self.ensure_turn().items.push(item);
    }

    fn handle_patch_apply_end(&mut self, payload: &PatchApplyEndEvent) {
        let status = if payload.success {
            PatchApplyStatus::Completed
        } else {
            PatchApplyStatus::Failed
        };
        let item = ThreadItem::FileChange {
            id: payload.call_id.clone(),
            changes: convert_patch_changes(&payload.changes),
            status,
        };
        self.ensure_turn().items.push(item);
    }

    fn handle_mcp_tool_call_end(&mut self, payload: &McpToolCallEndEvent) {
        let status = if payload.is_success() {
            McpToolCallStatus::Completed
        } else {
            McpToolCallStatus::Failed
        };
        let duration_ms = i64::try_from(payload.duration.as_millis()).ok();
        let (result, error) = match &payload.result {
            Ok(value) => (
                Some(McpToolCallResult {
                    content: value.content.clone(),
                    structured_content: value.structured_content.clone(),
                }),
                None,
            ),
            Err(message) => (
                None,
                Some(McpToolCallError {
                    message: message.clone(),
                }),
            ),
        };
        let item = ThreadItem::McpToolCall {
            id: payload.call_id.clone(),
            server: payload.invocation.server.clone(),
            tool: payload.invocation.tool.clone(),
            status,
            arguments: payload
                .invocation
                .arguments
                .clone()
                .unwrap_or(serde_json::Value::Null),
            result,
            error,
            duration_ms,
        };
        self.ensure_turn().items.push(item);
    }

    fn handle_view_image_tool_call(&mut self, payload: &ViewImageToolCallEvent) {
        let item = ThreadItem::ImageView {
            id: payload.call_id.clone(),
            path: payload.path.to_string_lossy().into_owned(),
        };
        self.ensure_turn().items.push(item);
    }

    fn handle_collab_agent_spawn_end(
        &mut self,
        payload: &codex_protocol::protocol::CollabAgentSpawnEndEvent,
    ) {
        let has_receiver = payload.new_thread_id.is_some();
        let status = match &payload.status {
            AgentStatus::Errored(_) | AgentStatus::NotFound => CollabAgentToolCallStatus::Failed,
            _ if has_receiver => CollabAgentToolCallStatus::Completed,
            _ => CollabAgentToolCallStatus::Failed,
        };
        let (receiver_thread_ids, agents_states) = match &payload.new_thread_id {
            Some(id) => {
                let receiver_id = id.to_string();
                let received_status = CollabAgentState::from(payload.status.clone());
                (
                    vec![receiver_id.clone()],
                    [(receiver_id, received_status)].into_iter().collect(),
                )
            }
            None => (Vec::new(), HashMap::new()),
        };
        self.ensure_turn()
            .items
            .push(ThreadItem::CollabAgentToolCall {
                id: payload.call_id.clone(),
                tool: CollabAgentTool::SpawnAgent,
                status,
                sender_thread_id: payload.sender_thread_id.to_string(),
                receiver_thread_ids,
                prompt: Some(payload.prompt.clone()),
                agents_states,
            });
    }

    fn handle_collab_agent_interaction_end(
        &mut self,
        payload: &codex_protocol::protocol::CollabAgentInteractionEndEvent,
    ) {
        let status = match &payload.status {
            AgentStatus::Errored(_) | AgentStatus::NotFound => CollabAgentToolCallStatus::Failed,
            _ => CollabAgentToolCallStatus::Completed,
        };
        let receiver_id = payload.receiver_thread_id.to_string();
        let received_status = CollabAgentState::from(payload.status.clone());
        self.ensure_turn()
            .items
            .push(ThreadItem::CollabAgentToolCall {
                id: payload.call_id.clone(),
                tool: CollabAgentTool::SendInput,
                status,
                sender_thread_id: payload.sender_thread_id.to_string(),
                receiver_thread_ids: vec![receiver_id.clone()],
                prompt: Some(payload.prompt.clone()),
                agents_states: [(receiver_id, received_status)].into_iter().collect(),
            });
    }

    fn handle_collab_waiting_end(
        &mut self,
        payload: &codex_protocol::protocol::CollabWaitingEndEvent,
    ) {
        let status = if payload
            .statuses
            .values()
            .any(|status| matches!(status, AgentStatus::Errored(_) | AgentStatus::NotFound))
        {
            CollabAgentToolCallStatus::Failed
        } else {
            CollabAgentToolCallStatus::Completed
        };
        let receiver_thread_ids = payload.statuses.keys().map(ToString::to_string).collect();
        let agents_states = payload
            .statuses
            .iter()
            .map(|(id, status)| (id.to_string(), CollabAgentState::from(status.clone())))
            .collect();
        self.ensure_turn()
            .items
            .push(ThreadItem::CollabAgentToolCall {
                id: payload.call_id.clone(),
                tool: CollabAgentTool::Wait,
                status,
                sender_thread_id: payload.sender_thread_id.to_string(),
                receiver_thread_ids,
                prompt: None,
                agents_states,
            });
    }

    fn handle_collab_close_end(&mut self, payload: &codex_protocol::protocol::CollabCloseEndEvent) {
        let status = match &payload.status {
            AgentStatus::Errored(_) | AgentStatus::NotFound => CollabAgentToolCallStatus::Failed,
            _ => CollabAgentToolCallStatus::Completed,
        };
        let receiver_id = payload.receiver_thread_id.to_string();
        let agents_states = [(
            receiver_id.clone(),
            CollabAgentState::from(payload.status.clone()),
        )]
        .into_iter()
        .collect();
        self.ensure_turn()
            .items
            .push(ThreadItem::CollabAgentToolCall {
                id: payload.call_id.clone(),
                tool: CollabAgentTool::CloseAgent,
                status,
                sender_thread_id: payload.sender_thread_id.to_string(),
                receiver_thread_ids: vec![receiver_id],
                prompt: None,
                agents_states,
            });
    }

    fn handle_context_compacted(&mut self, _payload: &ContextCompactedEvent) {
        let id = self.next_item_id();
        self.ensure_turn()
            .items
            .push(ThreadItem::ContextCompaction { id });
    }

    fn handle_entered_review_mode(&mut self, payload: &codex_protocol::protocol::ReviewRequest) {
        let review = payload
            .user_facing_hint
            .clone()
            .unwrap_or_else(|| "Review requested.".to_string());
        let id = self.next_item_id();
        self.ensure_turn()
            .items
            .push(ThreadItem::EnteredReviewMode { id, review });
    }

    fn handle_exited_review_mode(
        &mut self,
        payload: &codex_protocol::protocol::ExitedReviewModeEvent,
    ) {
        let review = payload
            .review_output
            .as_ref()
            .map(render_review_output_text)
            .unwrap_or_else(|| REVIEW_FALLBACK_MESSAGE.to_string());
        let id = self.next_item_id();
        self.ensure_turn()
            .items
            .push(ThreadItem::ExitedReviewMode { id, review });
    }

    fn handle_error(&mut self, payload: &ErrorEvent) {
        let turn = self.ensure_turn();
        turn.status = TurnStatus::Failed;
        turn.error = Some(V2TurnError {
            message: payload.message.clone(),
            codex_error_info: payload.codex_error_info.clone().map(Into::into),
            additional_details: None,
        });
    }

    fn handle_turn_aborted(&mut self, _payload: &TurnAbortedEvent) {
        let Some(turn) = self.current_turn.as_mut() else {
            return;
        };
        turn.status = TurnStatus::Interrupted;
    }

    fn handle_thread_rollback(&mut self, payload: &ThreadRolledBackEvent) {
        self.finish_current_turn();

        let n = usize::try_from(payload.num_turns).unwrap_or(usize::MAX);
        if n >= self.turns.len() {
            self.turns.clear();
        } else {
            self.turns.truncate(self.turns.len().saturating_sub(n));
        }

        // Re-number subsequent synthetic ids so the pruned history is consistent.
        self.next_turn_index =
            i64::try_from(self.turns.len().saturating_add(1)).unwrap_or(i64::MAX);
        let item_count: usize = self.turns.iter().map(|t| t.items.len()).sum();
        self.next_item_index = i64::try_from(item_count.saturating_add(1)).unwrap_or(i64::MAX);
    }

    fn finish_current_turn(&mut self) {
        if let Some(turn) = self.current_turn.take() {
            if turn.items.is_empty() {
                return;
            }
            self.turns.push(turn.into());
        }
    }

    fn new_turn(&mut self) -> PendingTurn {
        PendingTurn {
            id: self.next_turn_id(),
            items: Vec::new(),
            error: None,
            status: TurnStatus::Completed,
        }
    }

    fn ensure_turn(&mut self) -> &mut PendingTurn {
        if self.current_turn.is_none() {
            let turn = self.new_turn();
            return self.current_turn.insert(turn);
        }

        if let Some(turn) = self.current_turn.as_mut() {
            return turn;
        }

        unreachable!("current turn must exist after initialization");
    }

    fn next_turn_id(&mut self) -> String {
        let id = format!("turn-{}", self.next_turn_index);
        self.next_turn_index += 1;
        id
    }

    fn next_item_id(&mut self) -> String {
        let id = format!("item-{}", self.next_item_index);
        self.next_item_index += 1;
        id
    }

    fn build_user_inputs(&self, payload: &UserMessageEvent) -> Vec<UserInput> {
        let mut content = Vec::new();
        if !payload.message.trim().is_empty() {
            content.push(UserInput::Text {
                text: payload.message.clone(),
                text_elements: payload
                    .text_elements
                    .iter()
                    .cloned()
                    .map(Into::into)
                    .collect(),
            });
        }
        if let Some(images) = &payload.images {
            for image in images {
                content.push(UserInput::Image { url: image.clone() });
            }
        }
        for path in &payload.local_images {
            content.push(UserInput::LocalImage { path: path.clone() });
        }
        content
    }
}

const REVIEW_FALLBACK_MESSAGE: &str = "Reviewer failed to output a response.";

fn render_review_output_text(output: &ReviewOutputEvent) -> String {
    let explanation = output.overall_explanation.trim();
    if explanation.is_empty() {
        REVIEW_FALLBACK_MESSAGE.to_string()
    } else {
        explanation.to_string()
    }
}

fn convert_patch_changes(
    changes: &HashMap<std::path::PathBuf, codex_protocol::protocol::FileChange>,
) -> Vec<FileUpdateChange> {
    let mut converted: Vec<FileUpdateChange> = changes
        .iter()
        .map(|(path, change)| FileUpdateChange {
            path: path.to_string_lossy().into_owned(),
            kind: map_patch_change_kind(change),
            diff: format_file_change_diff(change),
        })
        .collect();
    converted.sort_by(|a, b| a.path.cmp(&b.path));
    converted
}

fn map_patch_change_kind(change: &codex_protocol::protocol::FileChange) -> PatchChangeKind {
    match change {
        codex_protocol::protocol::FileChange::Add { .. } => PatchChangeKind::Add,
        codex_protocol::protocol::FileChange::Delete { .. } => PatchChangeKind::Delete,
        codex_protocol::protocol::FileChange::Update { move_path, .. } => PatchChangeKind::Update {
            move_path: move_path.clone(),
        },
    }
}

fn format_file_change_diff(change: &codex_protocol::protocol::FileChange) -> String {
    match change {
        codex_protocol::protocol::FileChange::Add { content } => content.clone(),
        codex_protocol::protocol::FileChange::Delete { content } => content.clone(),
        codex_protocol::protocol::FileChange::Update {
            unified_diff,
            move_path,
        } => {
            if let Some(path) = move_path {
                format!("{unified_diff}\n\nMoved to: {}", path.display())
            } else {
                unified_diff.clone()
            }
        }
    }
}

struct PendingTurn {
    id: String,
    items: Vec<ThreadItem>,
    error: Option<TurnError>,
    status: TurnStatus,
}

impl From<PendingTurn> for Turn {
    fn from(value: PendingTurn) -> Self {
        Self {
            id: value.id,
            items: value.items,
            error: value.error,
            status: value.status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::WebSearchAction as CoreWebSearchAction;
    use codex_protocol::parse_command::ParsedCommand;
    use codex_protocol::protocol::AgentMessageEvent;
    use codex_protocol::protocol::AgentReasoningEvent;
    use codex_protocol::protocol::AgentReasoningRawContentEvent;
    use codex_protocol::protocol::ExecCommandEndEvent;
    use codex_protocol::protocol::ExecCommandSource;
    use codex_protocol::protocol::McpInvocation;
    use codex_protocol::protocol::McpToolCallEndEvent;
    use codex_protocol::protocol::ThreadRolledBackEvent;
    use codex_protocol::protocol::TurnAbortReason;
    use codex_protocol::protocol::TurnAbortedEvent;
    use codex_protocol::protocol::UserMessageEvent;
    use codex_protocol::protocol::WebSearchEndEvent;
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn builds_multiple_turns_with_reasoning_items() {
        let events = vec![
            EventMsg::UserMessage(UserMessageEvent {
                message: "First turn".into(),
                images: Some(vec!["https://example.com/one.png".into()]),
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "Hi there".into(),
            }),
            EventMsg::AgentReasoning(AgentReasoningEvent {
                text: "thinking".into(),
            }),
            EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent {
                text: "full reasoning".into(),
            }),
            EventMsg::UserMessage(UserMessageEvent {
                message: "Second turn".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "Reply two".into(),
            }),
        ];

        let turns = build_turns_from_event_msgs(&events);
        assert_eq!(turns.len(), 2);

        let first = &turns[0];
        assert_eq!(first.id, "turn-1");
        assert_eq!(first.status, TurnStatus::Completed);
        assert_eq!(first.items.len(), 3);
        assert_eq!(
            first.items[0],
            ThreadItem::UserMessage {
                id: "item-1".into(),
                content: vec![
                    UserInput::Text {
                        text: "First turn".into(),
                        text_elements: Vec::new(),
                    },
                    UserInput::Image {
                        url: "https://example.com/one.png".into(),
                    }
                ],
            }
        );
        assert_eq!(
            first.items[1],
            ThreadItem::AgentMessage {
                id: "item-2".into(),
                text: "Hi there".into(),
            }
        );
        assert_eq!(
            first.items[2],
            ThreadItem::Reasoning {
                id: "item-3".into(),
                summary: vec!["thinking".into()],
                content: vec!["full reasoning".into()],
            }
        );

        let second = &turns[1];
        assert_eq!(second.id, "turn-2");
        assert_eq!(second.items.len(), 2);
        assert_eq!(
            second.items[0],
            ThreadItem::UserMessage {
                id: "item-4".into(),
                content: vec![UserInput::Text {
                    text: "Second turn".into(),
                    text_elements: Vec::new(),
                }],
            }
        );
        assert_eq!(
            second.items[1],
            ThreadItem::AgentMessage {
                id: "item-5".into(),
                text: "Reply two".into(),
            }
        );
    }

    #[test]
    fn splits_reasoning_when_interleaved() {
        let events = vec![
            EventMsg::UserMessage(UserMessageEvent {
                message: "Turn start".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentReasoning(AgentReasoningEvent {
                text: "first summary".into(),
            }),
            EventMsg::AgentReasoningRawContent(AgentReasoningRawContentEvent {
                text: "first content".into(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "interlude".into(),
            }),
            EventMsg::AgentReasoning(AgentReasoningEvent {
                text: "second summary".into(),
            }),
        ];

        let turns = build_turns_from_event_msgs(&events);
        assert_eq!(turns.len(), 1);
        let turn = &turns[0];
        assert_eq!(turn.items.len(), 4);

        assert_eq!(
            turn.items[1],
            ThreadItem::Reasoning {
                id: "item-2".into(),
                summary: vec!["first summary".into()],
                content: vec!["first content".into()],
            }
        );
        assert_eq!(
            turn.items[3],
            ThreadItem::Reasoning {
                id: "item-4".into(),
                summary: vec!["second summary".into()],
                content: Vec::new(),
            }
        );
    }

    #[test]
    fn marks_turn_as_interrupted_when_aborted() {
        let events = vec![
            EventMsg::UserMessage(UserMessageEvent {
                message: "Please do the thing".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "Working...".into(),
            }),
            EventMsg::TurnAborted(TurnAbortedEvent {
                reason: TurnAbortReason::Replaced,
            }),
            EventMsg::UserMessage(UserMessageEvent {
                message: "Let's try again".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "Second attempt complete.".into(),
            }),
        ];

        let turns = build_turns_from_event_msgs(&events);
        assert_eq!(turns.len(), 2);

        let first_turn = &turns[0];
        assert_eq!(first_turn.status, TurnStatus::Interrupted);
        assert_eq!(first_turn.items.len(), 2);
        assert_eq!(
            first_turn.items[0],
            ThreadItem::UserMessage {
                id: "item-1".into(),
                content: vec![UserInput::Text {
                    text: "Please do the thing".into(),
                    text_elements: Vec::new(),
                }],
            }
        );
        assert_eq!(
            first_turn.items[1],
            ThreadItem::AgentMessage {
                id: "item-2".into(),
                text: "Working...".into(),
            }
        );

        let second_turn = &turns[1];
        assert_eq!(second_turn.status, TurnStatus::Completed);
        assert_eq!(second_turn.items.len(), 2);
        assert_eq!(
            second_turn.items[0],
            ThreadItem::UserMessage {
                id: "item-3".into(),
                content: vec![UserInput::Text {
                    text: "Let's try again".into(),
                    text_elements: Vec::new(),
                }],
            }
        );
        assert_eq!(
            second_turn.items[1],
            ThreadItem::AgentMessage {
                id: "item-4".into(),
                text: "Second attempt complete.".into(),
            }
        );
    }

    #[test]
    fn drops_last_turns_on_thread_rollback() {
        let events = vec![
            EventMsg::UserMessage(UserMessageEvent {
                message: "First".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "A1".into(),
            }),
            EventMsg::UserMessage(UserMessageEvent {
                message: "Second".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "A2".into(),
            }),
            EventMsg::ThreadRolledBack(ThreadRolledBackEvent { num_turns: 1 }),
            EventMsg::UserMessage(UserMessageEvent {
                message: "Third".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "A3".into(),
            }),
        ];

        let turns = build_turns_from_event_msgs(&events);
        let expected = vec![
            Turn {
                id: "turn-1".into(),
                status: TurnStatus::Completed,
                error: None,
                items: vec![
                    ThreadItem::UserMessage {
                        id: "item-1".into(),
                        content: vec![UserInput::Text {
                            text: "First".into(),
                            text_elements: Vec::new(),
                        }],
                    },
                    ThreadItem::AgentMessage {
                        id: "item-2".into(),
                        text: "A1".into(),
                    },
                ],
            },
            Turn {
                id: "turn-2".into(),
                status: TurnStatus::Completed,
                error: None,
                items: vec![
                    ThreadItem::UserMessage {
                        id: "item-3".into(),
                        content: vec![UserInput::Text {
                            text: "Third".into(),
                            text_elements: Vec::new(),
                        }],
                    },
                    ThreadItem::AgentMessage {
                        id: "item-4".into(),
                        text: "A3".into(),
                    },
                ],
            },
        ];
        assert_eq!(turns, expected);
    }

    #[test]
    fn thread_rollback_clears_all_turns_when_num_turns_exceeds_history() {
        let events = vec![
            EventMsg::UserMessage(UserMessageEvent {
                message: "One".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "A1".into(),
            }),
            EventMsg::UserMessage(UserMessageEvent {
                message: "Two".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::AgentMessage(AgentMessageEvent {
                message: "A2".into(),
            }),
            EventMsg::ThreadRolledBack(ThreadRolledBackEvent { num_turns: 99 }),
        ];

        let turns = build_turns_from_event_msgs(&events);
        assert_eq!(turns, Vec::<Turn>::new());
    }

    #[test]
    fn reconstructs_tool_items_from_persisted_completion_events() {
        let events = vec![
            EventMsg::UserMessage(UserMessageEvent {
                message: "run tools".into(),
                images: None,
                text_elements: Vec::new(),
                local_images: Vec::new(),
            }),
            EventMsg::WebSearchEnd(WebSearchEndEvent {
                call_id: "search-1".into(),
                query: "codex".into(),
                action: CoreWebSearchAction::Search {
                    query: Some("codex".into()),
                    queries: None,
                },
            }),
            EventMsg::ExecCommandEnd(ExecCommandEndEvent {
                call_id: "exec-1".into(),
                process_id: Some("pid-1".into()),
                turn_id: "turn-1".into(),
                command: vec!["echo".into(), "hello world".into()],
                cwd: PathBuf::from("/tmp"),
                parsed_cmd: vec![ParsedCommand::Unknown {
                    cmd: "echo hello world".into(),
                }],
                source: ExecCommandSource::Agent,
                interaction_input: None,
                stdout: String::new(),
                stderr: String::new(),
                aggregated_output: "hello world\n".into(),
                exit_code: 0,
                duration: Duration::from_millis(12),
                formatted_output: String::new(),
            }),
            EventMsg::McpToolCallEnd(McpToolCallEndEvent {
                call_id: "mcp-1".into(),
                invocation: McpInvocation {
                    server: "docs".into(),
                    tool: "lookup".into(),
                    arguments: Some(serde_json::json!({"id":"123"})),
                },
                duration: Duration::from_millis(8),
                result: Err("boom".into()),
            }),
        ];

        let turns = build_turns_from_event_msgs(&events);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].items.len(), 4);
        assert_eq!(
            turns[0].items[1],
            ThreadItem::WebSearch {
                id: "search-1".into(),
                query: "codex".into(),
                action: Some(WebSearchAction::Search {
                    query: Some("codex".into()),
                    queries: None,
                }),
            }
        );
        assert_eq!(
            turns[0].items[2],
            ThreadItem::CommandExecution {
                id: "exec-1".into(),
                command: "echo 'hello world'".into(),
                cwd: PathBuf::from("/tmp"),
                process_id: Some("pid-1".into()),
                status: CommandExecutionStatus::Completed,
                command_actions: vec![CommandAction::Unknown {
                    command: "echo hello world".into(),
                }],
                aggregated_output: Some("hello world\n".into()),
                exit_code: Some(0),
                duration_ms: Some(12),
            }
        );
        assert_eq!(
            turns[0].items[3],
            ThreadItem::McpToolCall {
                id: "mcp-1".into(),
                server: "docs".into(),
                tool: "lookup".into(),
                status: McpToolCallStatus::Failed,
                arguments: serde_json::json!({"id":"123"}),
                result: None,
                error: Some(McpToolCallError {
                    message: "boom".into(),
                }),
                duration_ms: Some(8),
            }
        );
    }
}
