#!/usr/bin/env bash
set -euo pipefail

CODEX_REPO="${CODEX_REPO:-/user/weihongliang/codex-wt-harness-local-2026-06-10}"
CARGO_BIN="${CARGO_BIN:-${HOME}/.cargo/bin/cargo}"

cd "${CODEX_REPO}/codex-rs"
"${CARGO_BIN}" build -p codex-cli --bin codex

echo "Built: ${CODEX_REPO}/codex-rs/target/debug/codex"
