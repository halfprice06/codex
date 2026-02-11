<p align="center"><code>npm i -g @openai/codex</code><br />or <code>brew install --cask codex</code></p>
<p align="center"><strong>Codex CLI</strong> is a coding agent from OpenAI that runs locally on your computer.
<p align="center">
  <img src="https://github.com/openai/codex/blob/main/.github/codex-cli-splash.png" alt="Codex CLI splash" width="80%" />
</p>
</br>
If you want Codex in your code editor (VS Code, Cursor, Windsurf), <a href="https://developers.openai.com/codex/ide">install in your IDE.</a>
</br>If you are looking for the <em>cloud-based agent</em> from OpenAI, <strong>Codex Web</strong>, go to <a href="https://chatgpt.com/codex">chatgpt.com/codex</a>.</p>

## Running this fork locally

If you are sharing a fork and want others to run your exact branch:

```shell
# 1) Clone your fork and checkout your branch
git clone https://github.com/<your-username>/codex.git
cd codex
git checkout <your-branch>

# 2) Build the CLI
cd codex-rs
cargo build -p codex-cli

# 3) Run it
./target/debug/codex
```

If you want a dedicated command name for this fork, add a shell alias:

```shell
# from codex-rs/ inside this repo
echo "alias codexrlm='$(pwd)/target/debug/codex'" >> ~/.zshrc
source ~/.zshrc

# now run your fork build with:
codexrlm
```

If you want the full setup (toolchain, dependencies, and packaging details), see [`docs/install.md`](./docs/install.md).

### Native RLM settings (no env vars required)

On first run, Codex writes missing `[native_rlm]` defaults into `~/.codex/config.toml`.
You can then tweak Native RLM directly there:

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
```

Environment variables (`CODEX_NATIVE_RLM*`) are still supported and will override
config values when set.

### Current approvals requirement

At the moment, Native RLM workflows expect full tool/file access. In the Codex TUI,
run `/approvals` and choose full access before testing RLM loops. If approvals are
stricter, tool execution and file editing may fail mid-loop.

---

## Quickstart

### Installing and running Codex CLI

Install globally with your preferred package manager:

```shell
# Install using npm
npm install -g @openai/codex
```

```shell
# Install using Homebrew
brew install --cask codex
```

Then simply run `codex` to get started.

<details>
<summary>You can also go to the <a href="https://github.com/openai/codex/releases/latest">latest GitHub Release</a> and download the appropriate binary for your platform.</summary>

Each GitHub Release contains many executables, but in practice, you likely want one of these:

- macOS
  - Apple Silicon/arm64: `codex-aarch64-apple-darwin.tar.gz`
  - x86_64 (older Mac hardware): `codex-x86_64-apple-darwin.tar.gz`
- Linux
  - x86_64: `codex-x86_64-unknown-linux-musl.tar.gz`
  - arm64: `codex-aarch64-unknown-linux-musl.tar.gz`

Each archive contains a single entry with the platform baked into the name (e.g., `codex-x86_64-unknown-linux-musl`), so you likely want to rename it to `codex` after extracting it.

</details>

### Using Codex with your ChatGPT plan

Run `codex` and select **Sign in with ChatGPT**. We recommend signing into your ChatGPT account to use Codex as part of your Plus, Pro, Team, Edu, or Enterprise plan. [Learn more about what's included in your ChatGPT plan](https://help.openai.com/en/articles/11369540-codex-in-chatgpt).

You can also use Codex with an API key, but this requires [additional setup](https://developers.openai.com/codex/auth#sign-in-with-an-api-key).

## Docs

- [**Codex Documentation**](https://developers.openai.com/codex)
- [**Contributing**](./docs/contributing.md)
- [**Installing & building**](./docs/install.md)
- [**Open source fund**](./docs/open-source-fund.md)

This repository is licensed under the [Apache-2.0 License](LICENSE).
