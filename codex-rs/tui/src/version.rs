/// The current Codex CLI version as embedded at compile time.
pub const CODEX_CLI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Branding shown in TUI header cards.
pub const CODEX_DISPLAY_NAME: &str = "OpenAI CodexRLM";

/// Version shown alongside TUI header card branding.
pub const CODEX_DISPLAY_VERSION: &str = "0.0.2";
