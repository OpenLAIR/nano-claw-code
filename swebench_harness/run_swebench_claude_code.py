#!/usr/bin/env python3
"""
SWE-bench evaluation harness for Claude Code with full trace logging.

Runs SWE-bench Lite or Verified using the Claude Code (Bun/TS) or
nano-claw-code (Python) agent in non-interactive (-p) mode. For each instance:
  1. Clone the repo at the base commit
  2. Invoke Claude Code with --output-format=stream-json --verbose
  3. Capture every intermediate event (tool calls, results, thinking, etc.)
  4. Extract the git diff as a patch
  5. Save predictions in SWE-bench JSONL format

Saved per instance (in results/traces/<instance_id>/):
  - raw_stream.jsonl       Full NDJSON stream from Claude Code (every event)
  - trace.json             Parsed structured trace (messages, tool calls, results)
  - patch.diff             The git diff produced
  - stderr.txt             stderr output from Claude Code
  - metadata.json          Timing, return code, token counts, etc.
  - input_prompt.txt       The exact prompt sent to Claude Code

Usage:
    python run_swebench_claude_code.py --max-instances 5
    python run_swebench_claude_code.py --model claude-sonnet-4-20250514
    python run_swebench_claude_code.py --evaluate --predictions results/predictions.jsonl

Environment:
    export ANTHROPIC_API_KEY='sk-ant-xxx'
    # or:
    export OPENROUTER_API_KEY='sk-or-xxx'
    export OPENROUTER_MODEL='anthropic/claude-3.7-sonnet'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLAUDE_CODE_DIR = SCRIPT_DIR.parent

# These are set from DEFAULT or overridden by --claude-code-dir at runtime
CLAUDE_CODE_DIR = DEFAULT_CLAUDE_CODE_DIR
CLAUDE_CODE_START = CLAUDE_CODE_DIR / "start.sh"
CLAUDE_CODE_ENTRYPOINT = CLAUDE_CODE_DIR / "src" / "entrypoints" / "cli.tsx"
CLAUDE_CODE_PRELOAD = CLAUDE_CODE_DIR / "preload.ts"


def is_nano_claw_python(agent_dir: Path) -> bool:
    """True if agent_dir is the Python nano-claw-code package (not Bun/TS Claude Code)."""
    return (agent_dir / "nano_claw_code" / "__main__.py").exists() and (
        agent_dir / "pyproject.toml"
    ).exists()


def set_claude_code_dir(path: Path) -> None:
    """Update all Claude Code paths to point to a different directory."""
    global CLAUDE_CODE_DIR, CLAUDE_CODE_START, CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE_PRELOAD
    CLAUDE_CODE_DIR = path.resolve()
    CLAUDE_CODE_START = CLAUDE_CODE_DIR / "start.sh"
    CLAUDE_CODE_ENTRYPOINT = CLAUDE_CODE_DIR / "src" / "entrypoints" / "cli.tsx"
    CLAUDE_CODE_PRELOAD = CLAUDE_CODE_DIR / "preload.ts"


def normalize_gateway_env(env: dict[str, str]) -> dict[str, str]:
    """Normalize OpenRouter convenience env vars into Anthropic-compatible ones."""
    normalized = dict(env)

    openrouter_key = normalized.get("OPENROUTER_API_KEY")
    openrouter_base = normalized.get("OPENROUTER_BASE_URL")
    openrouter_model = normalized.get("OPENROUTER_MODEL")

    if openrouter_key and not normalized.get("ANTHROPIC_AUTH_TOKEN"):
        normalized["ANTHROPIC_AUTH_TOKEN"] = openrouter_key

    if openrouter_base and not normalized.get("ANTHROPIC_BASE_URL"):
        normalized["ANTHROPIC_BASE_URL"] = openrouter_base

    if openrouter_model and not normalized.get("ANTHROPIC_MODEL"):
        normalized["ANTHROPIC_MODEL"] = openrouter_model

    if openrouter_model and not normalized.get("MODEL"):
        normalized["MODEL"] = openrouter_model

    if openrouter_key and not openrouter_base and not normalized.get("ANTHROPIC_BASE_URL"):
        normalized["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"

    return normalized

PROMPT_TEMPLATE = """\
I need you to fix a GitHub issue in this repository.

<issue>
{problem_statement}
</issue>

{hints_section}\
Instructions:
- The repository is already cloned and checked out at the correct base commit.
- Your current working directory is the repository root.
- Make the minimal changes necessary to fix the issue.
- Do NOT create new test files unless absolutely necessary.
- After making changes, verify them by running relevant tests if possible.
- When you are done, just say "DONE" — do not commit or push.\
"""


@dataclass
class HarnessConfig:
    dataset: str = "princeton-nlp/SWE-bench_Lite"
    split: str = "test"
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 50
    max_instances: int | None = None
    timeout_per_instance: int = 1800  # 30 minutes
    results_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "results")
    workspaces_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "workspaces")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    use_bare_mode: bool = False
    parallel: int = 1
    instance_ids: list[str] | None = None
    resume_from: str | None = None


# ─── Trace Parsing ────────────────────────────────────────────────────────────


def parse_stream_json_trace(raw_lines: list[str]) -> dict[str, Any]:
    """
    Parse the NDJSON stream from Claude Code (--output-format=stream-json --verbose)
    into a structured trace with all intermediate steps.

    The stream contains events like:
      - type=assistant   (model responses, may contain tool_use content blocks)
      - type=user        (tool results fed back to the model)
      - type=result      (final result with cost/duration info)
      - type=system      (hook events, init events)
      - type=stream_event (partial content deltas — real-time streaming)
    """
    events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    thinking_blocks: list[dict[str, Any]] = []
    system_events: list[dict[str, Any]] = []
    result_info: dict[str, Any] = {}
    parse_errors: list[dict[str, Any]] = []

    for line_num, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors.append({"line_num": line_num, "raw": line[:500]})
            continue

        events.append(event)
        event_type = event.get("type", "")

        if event_type == "assistant":
            msg_record = {
                "type": "assistant",
                "message_id": event.get("uuid", event.get("message", {}).get("id")),
                "model": event.get("message", {}).get("model"),
                "stop_reason": event.get("message", {}).get("stop_reason"),
                "usage": event.get("message", {}).get("usage"),
                "content": [],
            }
            for block in event.get("message", {}).get("content", event.get("content", [])):
                block_type = block.get("type", "")
                if block_type == "text":
                    msg_record["content"].append({
                        "type": "text",
                        "text": block.get("text", ""),
                    })
                elif block_type == "tool_use":
                    tool_call = {
                        "type": "tool_use",
                        "tool_use_id": block.get("id", ""),
                        "tool_name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }
                    msg_record["content"].append(tool_call)
                    tool_calls.append({
                        **tool_call,
                        "message_id": msg_record["message_id"],
                    })
                elif block_type == "thinking":
                    thinking = {
                        "type": "thinking",
                        "thinking": block.get("thinking", ""),
                    }
                    msg_record["content"].append(thinking)
                    thinking_blocks.append({
                        **thinking,
                        "message_id": msg_record["message_id"],
                    })
                else:
                    msg_record["content"].append(block)
            messages.append(msg_record)

        elif event_type == "user":
            msg_record = {
                "type": "user",
                "message_id": event.get("uuid"),
                "content": [],
            }
            for block in event.get("message", {}).get("content", event.get("content", [])):
                block_type = block.get("type", "")
                if block_type == "tool_result":
                    result_record = {
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id", ""),
                        "is_error": block.get("is_error", False),
                        "content": _extract_tool_result_content(block),
                    }
                    msg_record["content"].append(result_record)
                    tool_results.append({
                        **result_record,
                        "message_id": msg_record["message_id"],
                    })
                else:
                    msg_record["content"].append(block)
            messages.append(msg_record)

        elif event_type == "result":
            result_info = {
                "subtype": event.get("subtype"),
                "cost_usd": event.get("cost_usd"),
                "duration_ms": event.get("duration_ms"),
                "duration_api_ms": event.get("duration_api_ms"),
                "num_turns": event.get("num_turns"),
                "is_error": event.get("is_error"),
                "total_cost_usd": event.get("total_cost_usd"),
                "usage": event.get("usage"),
                "session_id": event.get("session_id"),
            }

        elif event_type == "system":
            system_events.append({
                "subtype": event.get("subtype"),
                "data": {k: v for k, v in event.items() if k not in ("type", "subtype")},
            })

    # Build the tool call -> result mapping
    tool_interactions = _build_tool_interaction_timeline(tool_calls, tool_results)

    return {
        "summary": {
            "total_events": len(events),
            "total_messages": len(messages),
            "total_tool_calls": len(tool_calls),
            "total_tool_results": len(tool_results),
            "total_thinking_blocks": len(thinking_blocks),
            "total_system_events": len(system_events),
            "parse_errors": len(parse_errors),
            "result": result_info,
        },
        "messages": messages,
        "tool_interactions": tool_interactions,
        "thinking_blocks": thinking_blocks,
        "system_events": system_events,
        "result": result_info,
        "parse_errors": parse_errors[:50],
    }


def _extract_tool_result_content(block: dict) -> str:
    """Extract readable content from a tool_result block."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(part)[:200])
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _build_tool_interaction_timeline(
    tool_calls: list[dict], tool_results: list[dict]
) -> list[dict[str, Any]]:
    """Match tool calls to their results by tool_use_id, preserving order."""
    results_by_id = {}
    for r in tool_results:
        results_by_id[r["tool_use_id"]] = r

    interactions = []
    for call in tool_calls:
        tid = call["tool_use_id"]
        interaction = {
            "tool_name": call["tool_name"],
            "tool_use_id": tid,
            "input": call["input"],
            "result": results_by_id.get(tid, {}).get("content", ""),
            "is_error": results_by_id.get(tid, {}).get("is_error", False),
        }
        interactions.append(interaction)

    return interactions


# ─── Dataset & Repo Management ────────────────────────────────────────────────


def load_dataset_instances(config: HarnessConfig) -> list[dict[str, Any]]:
    """Load SWE-bench instances from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading dataset: %s (split=%s)", config.dataset, config.split)
    dataset = load_dataset(config.dataset, split=config.split)
    instances = list(dataset)
    logger.info("Loaded %d instances", len(instances))
    return instances


def setup_repo(instance: dict[str, Any], workspaces_dir: Path) -> Path:
    """Clone the repo at the base commit. Returns workspace path."""
    instance_id: str = instance["instance_id"]
    repo: str = instance["repo"]
    base_commit: str = instance["base_commit"]

    safe_id = instance_id.replace("/", "__")
    workspace = workspaces_dir / safe_id

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    logger.info("  Cloning %s at %s", repo, base_commit[:10])

    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", base_commit],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "clean", "-fdx"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=workspace,
    )

    return workspace


def get_patch(workspace: Path) -> str:
    """Extract the git diff (staged + unstaged) from the workspace."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True,
        text=True,
        cwd=workspace,
    )
    patch = result.stdout
    if not patch.strip():
        return ""
    # Ensure patch ends with newline — required by the `patch` utility
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


# ─── Claude Code Invocation ──────────────────────────────────────────────────


def _find_bun() -> str:
    """Locate the bun executable."""
    candidates = [
        Path.home() / ".bun" / "bin" / "bun",
        Path("/usr/local/bin/bun"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("bun")
    if found:
        return found
    raise FileNotFoundError(
        "bun not found. Install it: curl -fsSL https://bun.sh/install | bash"
    )


def run_claude_code(
    prompt: str,
    workspace: Path,
    config: HarnessConfig,
    instance_id: str,
    trace_dir: Path,
) -> tuple[int, float]:
    """
    Invoke Claude Code in non-interactive mode with full stream-json tracing.

    IMPORTANT: We call bun directly (not start.sh) because start.sh does
    `cd "$DIR"` which overrides the cwd to start-claude-code's directory.
    We need cwd to be the cloned repo workspace so Claude Code operates
    on the correct repository.

    Streams stdout (NDJSON events) to trace_dir/raw_stream.jsonl in real time,
    so partial results are saved even if the process is killed or times out.

    Returns (returncode, elapsed_seconds).
    """
    bun = _find_bun()

    env = normalize_gateway_env(
        {
            **os.environ,
            "SWE_BENCH_RUN_ID": config.run_id,
            "SWE_BENCH_INSTANCE_ID": instance_id,
            "SWE_BENCH_TASK_ID": instance_id,
        }
    )

    if "ANTHROPIC_API_KEY" in env:
        env.update(
            {
                "ANTHROPIC_API_KEY": env["ANTHROPIC_API_KEY"],
            }
        )

    if "ANTHROPIC_BASE_URL" in env:
        env.setdefault("DISABLE_PROMPT_CACHING", "1")
        env.setdefault("DISABLE_INTERLEAVED_THINKING", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")

    # Auto-detect third-party proxy and disable incompatible features
    base_url = env.get("ANTHROPIC_BASE_URL", "")
    if base_url and "anthropic.com" not in base_url:
        env.setdefault("DISABLE_PROMPT_CACHING", "1")
        env.setdefault("DISABLE_INTERLEAVED_THINKING", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")

    cmd = [
        bun,
        "--preload", str(CLAUDE_CODE_PRELOAD),
        str(CLAUDE_CODE_ENTRYPOINT),
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--model", config.model,
        "--max-turns", str(config.max_turns),
        "--output-format", "stream-json",
        "--verbose",
    ]

    if config.use_bare_mode:
        cmd.append("--bare")

    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_stream_path = trace_dir / "raw_stream.jsonl"
    stderr_path = trace_dir / "stderr.txt"

    start = time.time()
    returncode = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workspace),
            env=env,
            stdin=subprocess.DEVNULL,
        )

        # Stream stdout to file line-by-line for crash resilience
        with open(raw_stream_path, "w") as stream_f:
            assert proc.stdout is not None
            for raw_line in iter(proc.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                stream_f.write(line)
                stream_f.flush()

                elapsed_so_far = time.time() - start
                if elapsed_so_far > config.timeout_per_instance:
                    logger.warning(
                        "  TIMEOUT after %.0fs for %s — killing process",
                        elapsed_so_far,
                        instance_id,
                    )
                    proc.kill()
                    break

        proc.wait(timeout=60)
        returncode = proc.returncode

        assert proc.stderr is not None
        stderr_content = proc.stderr.read().decode("utf-8", errors="replace")
        stderr_path.write_text(stderr_content)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        logger.warning("  TIMEOUT (wait) for %s", instance_id)
    except Exception as e:
        logger.error("  Process error for %s: %s", instance_id, e)
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass

    elapsed = time.time() - start
    return returncode, elapsed


def run_nano_claw_python(
    prompt: str,
    workspace: Path,
    config: HarnessConfig,
    instance_id: str,
    trace_dir: Path,
) -> tuple[int, float]:
    """
    Invoke nano-claw-code (Python) with the same flags and stream-json tracing
    contract as run_claude_code().
    """
    env = normalize_gateway_env(
        {
            **os.environ,
            "SWE_BENCH_RUN_ID": config.run_id,
            "SWE_BENCH_INSTANCE_ID": instance_id,
            "SWE_BENCH_TASK_ID": instance_id,
            "PYTHONPATH": str(CLAUDE_CODE_DIR),
            "PYTHONUNBUFFERED": "1",
        }
    )

    if "ANTHROPIC_API_KEY" in env:
        env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_API_KEY"]

    if "ANTHROPIC_BASE_URL" in env:
        env.setdefault("DISABLE_PROMPT_CACHING", "1")
        env.setdefault("DISABLE_INTERLEAVED_THINKING", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")

    base_url = env.get("ANTHROPIC_BASE_URL", "")
    if base_url and "anthropic.com" not in base_url:
        env.setdefault("DISABLE_PROMPT_CACHING", "1")
        env.setdefault("DISABLE_INTERLEAVED_THINKING", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")

    cmd = [
        sys.executable,
        "-m",
        "nano_claw_code",
        "-p",
        prompt,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--model",
        config.model,
        "--max-turns",
        str(config.max_turns),
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if config.use_bare_mode:
        cmd.append("--bare")

    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_stream_path = trace_dir / "raw_stream.jsonl"
    stderr_path = trace_dir / "stderr.txt"

    start = time.time()
    returncode = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workspace),
            env=env,
            stdin=subprocess.DEVNULL,
        )

        with open(raw_stream_path, "w") as stream_f:
            assert proc.stdout is not None
            for raw_line in iter(proc.stdout.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                stream_f.write(line)
                stream_f.flush()

                elapsed_so_far = time.time() - start
                if elapsed_so_far > config.timeout_per_instance:
                    logger.warning(
                        "  TIMEOUT after %.0fs for %s — killing process",
                        elapsed_so_far,
                        instance_id,
                    )
                    proc.kill()
                    break

        proc.wait(timeout=60)
        returncode = proc.returncode

        assert proc.stderr is not None
        stderr_content = proc.stderr.read().decode("utf-8", errors="replace")
        stderr_path.write_text(stderr_content)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        logger.warning("  TIMEOUT (wait) for %s", instance_id)
    except Exception as e:
        logger.error("  Process error for %s: %s", instance_id, e)
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass

    elapsed = time.time() - start
    return returncode, elapsed


def run_agent_framework(
    prompt: str,
    workspace: Path,
    config: HarnessConfig,
    instance_id: str,
    trace_dir: Path,
) -> tuple[int, float]:
    """Run Bun Claude Code or Python nano-claw-code depending on --claude-code-dir."""
    if is_nano_claw_python(CLAUDE_CODE_DIR):
        logger.info("  Agent backend: nano-claw-code (Python)")
        return run_nano_claw_python(prompt, workspace, config, instance_id, trace_dir)
    return run_claude_code(prompt, workspace, config, instance_id, trace_dir)


# ─── Prediction Management ────────────────────────────────────────────────────


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing predictions keyed by instance_id."""
    if not path.exists():
        return {}
    existing = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pred = json.loads(line)
                existing[pred["instance_id"]] = pred
    return existing


def save_predictions(predictions: list[dict[str, Any]], path: Path) -> None:
    """Save predictions as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


# ─── Per-Instance Runner ─────────────────────────────────────────────────────


def run_single_instance(
    instance: dict[str, Any],
    config: HarnessConfig,
) -> dict[str, Any]:
    """Run Claude Code on a single SWE-bench instance. Returns prediction dict."""
    instance_id = instance["instance_id"]
    safe_id = instance_id.replace("/", "__")
    logger.info("Processing: %s", instance_id)

    trace_dir = config.results_dir / "traces" / safe_id
    instance_start_time = time.time()
    instance_start_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Clone repo
    clone_start = time.time()
    try:
        workspace = setup_repo(instance, config.workspaces_dir)
    except Exception as e:
        logger.error("  Failed to clone: %s", e)
        return {
            "instance_id": instance_id,
            "model_patch": "",
            "model_name_or_path": config.model,
        }
    clone_seconds = round(time.time() - clone_start, 2)
    logger.info("  Clone: %.1fs", clone_seconds)

    # 2. Build prompt
    hints = instance.get("hints_text", "")
    hints_section = f"Hints from the maintainers:\n{hints}\n\n" if hints else ""

    prompt = PROMPT_TEMPLATE.format(
        problem_statement=instance["problem_statement"],
        hints_section=hints_section,
    )

    # Save the input prompt
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "input_prompt.txt").write_text(prompt)

    # Save instance metadata from SWE-bench
    (trace_dir / "instance_info.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "repo": instance.get("repo"),
                "base_commit": instance.get("base_commit"),
                "problem_statement": instance.get("problem_statement"),
                "hints_text": instance.get("hints_text", ""),
                "created_at": instance.get("created_at"),
                "version": instance.get("version"),
            },
            indent=2,
            default=str,
        )
    )

    # 3. Run Claude Code with full streaming trace
    agent_start = time.time()
    returncode, agent_seconds = run_agent_framework(
        prompt=prompt,
        workspace=workspace,
        config=config,
        instance_id=instance_id,
        trace_dir=trace_dir,
    )
    agent_seconds = round(agent_seconds, 2)

    # 4. Extract patch
    patch_start = time.time()
    patch = get_patch(workspace)
    patch_seconds = round(time.time() - patch_start, 2)

    total_seconds = round(time.time() - instance_start_time, 2)
    instance_end_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        "  Done: rc=%d, total=%.0fs (clone=%.0fs, agent=%.0fs), patch=%d chars",
        returncode,
        total_seconds,
        clone_seconds,
        agent_seconds,
        len(patch),
    )

    # 5. Save patch
    (trace_dir / "patch.diff").write_text(patch)

    # 6. Parse the raw stream into a structured trace
    raw_stream_path = trace_dir / "raw_stream.jsonl"
    raw_lines = []
    if raw_stream_path.exists():
        raw_lines = raw_stream_path.read_text().splitlines()

    trace = parse_stream_json_trace(raw_lines)

    # Timing breakdown
    timing = {
        "instance_id": instance_id,
        "start_time": instance_start_ts,
        "end_time": instance_end_ts,
        "total_seconds": total_seconds,
        "clone_seconds": clone_seconds,
        "agent_seconds": agent_seconds,
        "patch_extract_seconds": patch_seconds,
        "timed_out": returncode == -1 or agent_seconds >= config.timeout_per_instance,
    }

    # Add harness-level metadata to the trace
    trace["harness_metadata"] = {
        "instance_id": instance_id,
        "run_id": config.run_id,
        "model": config.model,
        "max_turns": config.max_turns,
        "returncode": returncode,
        "timing": timing,
        "patch_chars": len(patch),
        "timeout_per_instance": config.timeout_per_instance,
        "timestamp": instance_end_ts,
    }

    (trace_dir / "trace.json").write_text(json.dumps(trace, indent=2, default=str))

    # 7. Save a compact metadata file for quick analysis
    (trace_dir / "metadata.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "returncode": returncode,
                "timing": timing,
                "patch_chars": len(patch),
                "total_events": trace["summary"]["total_events"],
                "total_messages": trace["summary"]["total_messages"],
                "total_tool_calls": trace["summary"]["total_tool_calls"],
                "tool_names_used": list(
                    {tc["tool_name"] for tc in trace.get("tool_interactions", [])}
                ),
                "cost_usd": trace.get("result", {}).get("total_cost_usd"),
                "num_turns": trace.get("result", {}).get("num_turns"),
                "timed_out": timing["timed_out"],
            },
            indent=2,
        )
    )

    # 8. Append to consolidated timing log (one line per instance)
    timing_log_path = config.results_dir / "timing.jsonl"
    with open(timing_log_path, "a") as f:
        timing_record = {
            **timing,
            "returncode": returncode,
            "patch_chars": len(patch),
            "cost_usd": trace.get("result", {}).get("total_cost_usd"),
            "num_turns": trace.get("result", {}).get("num_turns"),
            "total_tool_calls": trace["summary"]["total_tool_calls"],
            "model": config.model,
            "run_id": config.run_id,
        }
        f.write(json.dumps(timing_record) + "\n")

    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": config.model,
    }


# ─── Main Evaluation Loop ────────────────────────────────────────────────────


def _load_instance_ids(raw: str) -> list[str]:
    """Parse --instance-ids value: file path (one ID per line) or comma-separated."""
    path = Path(raw)
    if path.is_file():
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def run_evaluation(config: HarnessConfig) -> list[dict[str, Any]]:
    """Run Claude Code on all SWE-bench instances and save predictions."""
    instances = load_dataset_instances(config)

    if config.instance_ids:
        id_set = set(config.instance_ids)
        instances = [inst for inst in instances if inst["instance_id"] in id_set]
        missing = id_set - {inst["instance_id"] for inst in instances}
        if missing:
            logger.warning("Instance IDs not found in dataset: %s", missing)
        logger.info("Filtered to %d instances by --instance-ids", len(instances))

    if config.max_instances:
        instances = instances[: config.max_instances]

    predictions_path = config.results_dir / "predictions.jsonl"
    existing = load_existing_predictions(predictions_path)

    if config.resume_from:
        found = False
        for i, inst in enumerate(instances):
            if inst["instance_id"] == config.resume_from:
                instances = instances[i:]
                found = True
                break
        if not found:
            logger.warning(
                "Resume target %s not found, starting from beginning",
                config.resume_from,
            )
    elif existing:
        completed_ids = {iid for iid, p in existing.items() if p.get("model_patch")}
        remaining = [
            inst for inst in instances if inst["instance_id"] not in completed_ids
        ]
        if len(remaining) < len(instances):
            logger.info(
                "Resuming: %d/%d already completed, %d remaining",
                len(instances) - len(remaining),
                len(instances),
                len(remaining),
            )
            instances = remaining

    predictions = list(existing.values())
    total = len(instances)

    logger.info("Run ID: %s", config.run_id)
    logger.info("Model: %s", config.model)
    logger.info("Max turns: %d", config.max_turns)
    logger.info("Timeout per instance: %ds", config.timeout_per_instance)
    logger.info("Parallel workers: %d", config.parallel)
    logger.info("Instances to process: %d", total)
    logger.info(
        "Trace output: %s", config.results_dir / "traces" / "<instance_id>"
    )

    if config.parallel <= 1:
        # Serial execution (original path)
        for i, instance in enumerate(instances):
            logger.info("=" * 60)
            logger.info("[%d/%d] %s", i + 1, total, instance["instance_id"])
            logger.info("=" * 60)

            pred = run_single_instance(instance, config)
            _merge_prediction(predictions, pred)
            save_predictions(predictions, predictions_path)
            logger.info(
                "  Saved %d predictions to %s", len(predictions), predictions_path
            )
    else:
        # Parallel execution
        lock = threading.Lock()
        completed_count = [0]

        def _worker(instance: dict[str, Any]) -> dict[str, Any]:
            iid = instance["instance_id"]
            with lock:
                completed_count[0] += 1
                idx = completed_count[0]
            logger.info("[%d/%d] START  %s", idx, total, iid)
            pred = run_single_instance(instance, config)
            with lock:
                _merge_prediction(predictions, pred)
                save_predictions(predictions, predictions_path)
            logger.info("[%d/%d] DONE   %s  patch=%d chars",
                        idx, total, iid, len(pred.get("model_patch", "")))
            return pred

        with ThreadPoolExecutor(max_workers=config.parallel) as pool:
            futures = {pool.submit(_worker, inst): inst for inst in instances}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    iid = futures[future]["instance_id"]
                    logger.error("Worker failed for %s: %s", iid, e)

    return predictions


def _merge_prediction(
    predictions: list[dict[str, Any]], pred: dict[str, Any]
) -> None:
    """Insert or replace a prediction in the list (by instance_id)."""
    for j, existing_pred in enumerate(predictions):
        if existing_pred["instance_id"] == pred["instance_id"]:
            predictions[j] = pred
            return
    predictions.append(pred)


def run_swebench_evaluation(
    predictions_path: Path,
    dataset: str,
    run_id: str,
    report_dir: Path,
) -> None:
    """Run the official SWE-bench Docker-based evaluation harness."""
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path", str(predictions_path),
        "--dataset_name", dataset,
        "--run_id", run_id,
        "--report_dir", str(report_dir),
    ]

    logger.info("Running SWE-bench evaluation:")
    logger.info("  %s", " ".join(cmd))
    logger.info("")
    logger.info("NOTE: This requires Docker. Make sure Docker is running.")

    subprocess.run(cmd, check=True)


# ─── Timing Utilities ─────────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_timing_summary(timing_log_path: Path, run_total_seconds: float) -> dict:
    """Build a timing summary from the consolidated timing.jsonl log."""
    records = []
    if timing_log_path.exists():
        with open(timing_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        return {"instance_count": 0, "run_total_seconds": run_total_seconds}

    def _stats(values: list[float]) -> dict:
        values = sorted(values)
        n = len(values)
        return {
            "min": round(values[0], 2),
            "max": round(values[-1], 2),
            "mean": round(sum(values) / n, 2),
            "median": round(values[n // 2], 2),
            "total": round(sum(values), 2),
        }

    return {
        "instance_count": len(records),
        "run_total_seconds": run_total_seconds,
        "run_total_human": _format_duration(run_total_seconds),
        "total_seconds": _stats([r["total_seconds"] for r in records]),
        "clone_seconds": _stats([r["clone_seconds"] for r in records]),
        "agent_seconds": _stats([r["agent_seconds"] for r in records]),
        "timed_out_count": sum(1 for r in records if r.get("timed_out")),
        "total_cost_usd": round(
            sum(r.get("cost_usd") or 0 for r in records), 4
        ),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using Claude Code agent (with full trace logging)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quick test (5 instances)
  python run_swebench_claude_code.py --max-instances 5

  # Full SWE-bench Lite run
  python run_swebench_claude_code.py --model claude-sonnet-4-20250514

  # SWE-bench Verified with Opus
  python run_swebench_claude_code.py \\
    --dataset princeton-nlp/SWE-bench_Verified \\
    --model claude-opus-4-20250514

  # Only evaluate existing predictions
  python run_swebench_claude_code.py \\
    --evaluate --predictions results/predictions.jsonl

Per-instance traces are saved to:
  results/traces/<instance_id>/
    raw_stream.jsonl     Full NDJSON stream (every event from Claude Code)
    trace.json           Parsed structured trace
    input_prompt.txt     Exact prompt sent
    instance_info.json   SWE-bench instance metadata
    patch.diff           Generated patch
    stderr.txt           stderr output
    metadata.json        Quick-reference metadata
""",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset (default: SWE-bench_Lite). "
        "Use 'princeton-nlp/SWE-bench_Verified' for the verified subset.",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Max agentic turns per instance (default: 50)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit the number of instances (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per instance in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: results/<variant_name>)",
    )
    parser.add_argument(
        "--workspaces-dir",
        type=Path,
        default=None,
        help="Directory for cloned repos (default: workspaces/<variant_name>)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for prediction generation (default: 1)",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        default=None,
        help="Filter to specific instances: file path (one ID per line) or comma-separated list",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Use --bare mode (skip hooks, LSP, plugins — faster but less features)",
    )
    parser.add_argument(
        "--claude-code-dir",
        type=Path,
        default=None,
        help="Path to agent directory (default: repo root, auto-detected as nano-claw-code).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a specific instance_id",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run SWE-bench evaluation harness on predictions",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions JSONL (for --evaluate)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.claude_code_dir:
        set_claude_code_dir(args.claude_code_dir)

    # Derive variant name from the claude-code directory for result isolation
    variant_name = CLAUDE_CODE_DIR.name  # e.g. "start-claude-code" or "nano-claude-code"
    logger.info("Claude Code variant: %s (%s)", variant_name, CLAUDE_CODE_DIR)

    if args.results_dir is None:
        args.results_dir = SCRIPT_DIR / "results" / variant_name
    if args.workspaces_dir is None:
        args.workspaces_dir = SCRIPT_DIR / "workspaces" / variant_name

    if not args.evaluate:
        normalized_env = normalize_gateway_env(dict(os.environ))
        if (
            "ANTHROPIC_API_KEY" not in normalized_env
            and "ANTHROPIC_AUTH_TOKEN" not in normalized_env
        ):
            logger.error(
                "No API key detected. Export one before running:\n"
                "  export ANTHROPIC_API_KEY='sk-ant-xxx'\n"
                "  # or\n"
                "  export OPENROUTER_API_KEY='sk-or-xxx'"
            )
            sys.exit(1)

        if is_nano_claw_python(CLAUDE_CODE_DIR):
            logger.info("Using nano-claw-code (Python); PYTHONPATH=%s", CLAUDE_CODE_DIR)
            try:
                import anthropic  # noqa: F401
            except ImportError:
                logger.error(
                    "nano-claw-code needs the anthropic package. Run:\n"
                    "  pip install -e %s",
                    CLAUDE_CODE_DIR,
                )
                sys.exit(1)
        else:
            if not CLAUDE_CODE_ENTRYPOINT.exists():
                logger.error(
                    "Claude Code / nano-claw not found at %s\n"
                    "Expected either:\n"
                    "  start-claude-code/  with src/entrypoints/cli.tsx\n"
                    "  nano-claw-code/     with nano_claw_code/__main__.py and pyproject.toml",
                    CLAUDE_CODE_DIR,
                )
                sys.exit(1)

            node_modules = CLAUDE_CODE_DIR / "node_modules" / "@anthropic-ai" / "sdk"
            if not node_modules.exists():
                logger.error(
                    "start-claude-code not set up. Run first:\n"
                    "  cd %s && node scripts/setup.mjs",
                    CLAUDE_CODE_DIR,
                )
                sys.exit(1)

            try:
                bun = _find_bun()
                logger.info("Using bun: %s", bun)
            except FileNotFoundError as e:
                logger.error("%s", e)
                sys.exit(1)

    if args.evaluate:
        pred_path = args.predictions or (args.results_dir / "predictions.jsonl")
        if not pred_path.exists():
            logger.error("Predictions file not found: %s", pred_path)
            sys.exit(1)
        run_swebench_evaluation(
            predictions_path=pred_path,
            dataset=args.dataset,
            run_id=f"{variant_name}-swebench",
            report_dir=args.results_dir / "swebench_eval_reports",
        )
        return

    instance_ids = None
    if args.instance_ids:
        instance_ids = _load_instance_ids(args.instance_ids)
        logger.info("Loaded %d instance IDs from filter", len(instance_ids))

    config = HarnessConfig(
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        max_turns=args.max_turns,
        max_instances=args.max_instances,
        timeout_per_instance=args.timeout,
        results_dir=args.results_dir,
        workspaces_dir=args.workspaces_dir,
        use_bare_mode=args.bare,
        parallel=args.parallel,
        instance_ids=instance_ids,
        resume_from=args.resume_from,
    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.workspaces_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = config.results_dir / f"run_config_{config.run_id}.json"
    run_config_path.write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "dataset": config.dataset,
                "split": config.split,
                "model": config.model,
                "max_turns": config.max_turns,
                "max_instances": config.max_instances,
                "timeout_per_instance": config.timeout_per_instance,
                "use_bare_mode": config.use_bare_mode,
                "parallel": config.parallel,
                "instance_ids_count": len(config.instance_ids) if config.instance_ids else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "trace_format": "stream-json (NDJSON)",
            },
            indent=2,
        )
    )

    run_start_time = time.time()
    predictions = run_evaluation(config)
    run_total_seconds = round(time.time() - run_start_time, 2)

    total = len(predictions)
    with_patch = sum(1 for p in predictions if p["model_patch"])

    # Generate timing summary from the consolidated log
    timing_log_path = config.results_dir / "timing.jsonl"
    timing_summary = _build_timing_summary(timing_log_path, run_total_seconds)
    summary_path = config.results_dir / "timing_summary.json"
    summary_path.write_text(json.dumps(timing_summary, indent=2))

    logger.info("=" * 60)
    logger.info("COMPLETE: %d/%d instances produced a patch", with_patch, total)
    logger.info("Total wall-clock time: %s", _format_duration(run_total_seconds))
    if timing_summary["instance_count"] > 0:
        logger.info(
            "Per-instance: avg=%.0fs, median=%.0fs, min=%.0fs, max=%.0fs",
            timing_summary["agent_seconds"]["mean"],
            timing_summary["agent_seconds"]["median"],
            timing_summary["agent_seconds"]["min"],
            timing_summary["agent_seconds"]["max"],
        )
    logger.info("Predictions: %s", config.results_dir / "predictions.jsonl")
    logger.info("Timing log:  %s", timing_log_path)
    logger.info("Timing summary: %s", summary_path)
    logger.info("")
    logger.info("Next step — run official SWE-bench evaluation:")
    logger.info(
        "  python run_swebench_claude_code.py --evaluate "
        "--predictions %s --dataset %s",
        config.results_dir / "predictions.jsonl",
        config.dataset,
    )


if __name__ == "__main__":
    main()
