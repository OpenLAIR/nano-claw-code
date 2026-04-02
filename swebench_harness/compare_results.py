#!/usr/bin/env python3
"""
Compare SWE-bench results between two agent variants.

Reads predictions, timing logs, and evaluation reports from two result
directories and prints a side-by-side comparison table.

Usage:
    python compare_results.py \
        --a results/nano-claw-code \
        --b results/start-claude-code \
        --label-a "nano-claw-code" \
        --label-b "start-claude-code"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_predictions(results_dir: Path) -> dict[str, dict[str, Any]]:
    path = results_dir / "predictions.jsonl"
    if not path.exists():
        return {}
    preds = {}
    for line in path.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            preds[p["instance_id"]] = p
    return preds


def load_eval_report(results_dir: Path) -> dict[str, Any]:
    """Find and load the SWE-bench evaluation JSON report."""
    for p in results_dir.rglob("*.json"):
        if "swebench" in p.name and p.name.endswith(".json"):
            try:
                data = json.loads(p.read_text())
                if "resolved_ids" in data:
                    return data
            except (json.JSONDecodeError, KeyError):
                continue
    # Also check parent directory (report sometimes written to harness root)
    harness_dir = results_dir.parent.parent
    for p in harness_dir.glob("*.swebench.json"):
        try:
            data = json.loads(p.read_text())
            if "resolved_ids" in data:
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    for p in harness_dir.glob("*.nano-claw-code-swebench.json"):
        try:
            data = json.loads(p.read_text())
            if "resolved_ids" in data:
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    return {}


def load_timing(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / "timing.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_token_usage(results_dir: Path) -> dict[str, dict[str, int]]:
    """Extract per-instance token usage from raw_stream.jsonl files."""
    traces_dir = results_dir / "traces"
    if not traces_dir.exists():
        return {}
    usage = {}
    for trace_dir in traces_dir.iterdir():
        if not trace_dir.is_dir():
            continue
        iid = trace_dir.name.replace("__", "/", 1)
        stream_path = trace_dir / "raw_stream.jsonl"
        if not stream_path.exists():
            continue
        total_input = 0
        total_output = 0
        total_cache_create = 0
        total_cache_read = 0
        for line in stream_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "assistant":
                u = event.get("message", {}).get("usage", {})
                if u:
                    total_input += u.get("input_tokens", 0)
                    total_output += u.get("output_tokens", 0)
                    total_cache_create += u.get("cache_creation_input_tokens", 0)
                    total_cache_read += u.get("cache_read_input_tokens", 0)
        usage[iid] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_creation_tokens": total_cache_create,
            "cache_read_tokens": total_cache_read,
            "total_input": total_input + total_cache_create + total_cache_read,
        }
    return usage


def _repo_from_id(instance_id: str) -> str:
    parts = instance_id.split("__")
    if len(parts) >= 2:
        return parts[0].replace("_", "-", 1) if "_" in parts[0] else parts[0]
    return instance_id


def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def compare(dir_a: Path, dir_b: Path, label_a: str, label_b: str) -> None:
    preds_a = load_predictions(dir_a)
    preds_b = load_predictions(dir_b)
    eval_a = load_eval_report(dir_a)
    eval_b = load_eval_report(dir_b)
    timing_a = {r["instance_id"]: r for r in load_timing(dir_a)}
    timing_b = {r["instance_id"]: r for r in load_timing(dir_b)}
    tokens_a = load_token_usage(dir_a)
    tokens_b = load_token_usage(dir_b)

    resolved_a = set(eval_a.get("resolved_ids", []))
    resolved_b = set(eval_b.get("resolved_ids", []))

    all_ids = sorted(set(preds_a) | set(preds_b))

    # ── Header ──
    w = max(len(label_a), len(label_b), 30)
    print("=" * 72)
    print(f"  SWE-bench Comparison: {label_a}  vs  {label_b}")
    print("=" * 72)
    print()

    # ── Overall Stats ──
    print("── Overall ──")
    print(f"  {'Metric':<25s}  {label_a:>{w}s}  {label_b:>{w}s}")
    print(f"  {'─' * 25}  {'─' * w}  {'─' * w}")

    n_a, n_b = len(preds_a), len(preds_b)
    patch_a = sum(1 for p in preds_a.values() if p.get("model_patch"))
    patch_b = sum(1 for p in preds_b.values() if p.get("model_patch"))

    print(f"  {'Instances submitted':<25s}  {n_a:>{w}d}  {n_b:>{w}d}")
    print(f"  {'Patches generated':<25s}  {_fmt_pct(patch_a, n_a):>{w}s}  {_fmt_pct(patch_b, n_b):>{w}s}")

    if resolved_a or resolved_b:
        ra, rb = len(resolved_a), len(resolved_b)
        print(f"  {'Resolved (passed tests)':<25s}  {_fmt_pct(ra, n_a):>{w}s}  {_fmt_pct(rb, n_b):>{w}s}")
    else:
        print(f"  {'Resolved (passed tests)':<25s}  {'(no eval)':>{w}s}  {'(no eval)':>{w}s}")

    # Timing
    if timing_a or timing_b:
        def _avg(records: dict, key: str) -> float:
            vals = [r[key] for r in records.values() if key in r]
            return sum(vals) / len(vals) if vals else 0

        avg_agent_a = _avg(timing_a, "agent_seconds")
        avg_agent_b = _avg(timing_b, "agent_seconds")
        avg_total_a = _avg(timing_a, "total_seconds")
        avg_total_b = _avg(timing_b, "total_seconds")

        print(f"  {'Avg agent time (s)':<25s}  {avg_agent_a:>{w}.1f}  {avg_agent_b:>{w}.1f}")
        print(f"  {'Avg total time (s)':<25s}  {avg_total_a:>{w}.1f}  {avg_total_b:>{w}.1f}")

        timeout_a = sum(1 for r in timing_a.values() if r.get("timed_out"))
        timeout_b = sum(1 for r in timing_b.values() if r.get("timed_out"))
        print(f"  {'Timeouts':<25s}  {timeout_a:>{w}d}  {timeout_b:>{w}d}")

    # Tokens
    if tokens_a or tokens_b:
        def _sum_field(tok: dict, field: str) -> int:
            return sum(t.get(field, 0) for t in tok.values())

        ti_a = _sum_field(tokens_a, "total_input")
        ti_b = _sum_field(tokens_b, "total_input")
        to_a = _sum_field(tokens_a, "output_tokens")
        to_b = _sum_field(tokens_b, "output_tokens")

        print(f"  {'Total input tokens':<25s}  {_fmt_tokens(ti_a):>{w}s}  {_fmt_tokens(ti_b):>{w}s}")
        print(f"  {'Total output tokens':<25s}  {_fmt_tokens(to_a):>{w}s}  {_fmt_tokens(to_b):>{w}s}")

        if n_a > 0 and tokens_a:
            avg_in_a = ti_a // max(len(tokens_a), 1)
            avg_out_a = to_a // max(len(tokens_a), 1)
        else:
            avg_in_a = avg_out_a = 0
        if n_b > 0 and tokens_b:
            avg_in_b = ti_b // max(len(tokens_b), 1)
            avg_out_b = to_b // max(len(tokens_b), 1)
        else:
            avg_in_b = avg_out_b = 0

        print(f"  {'Avg input tokens/task':<25s}  {_fmt_tokens(avg_in_a):>{w}s}  {_fmt_tokens(avg_in_b):>{w}s}")
        print(f"  {'Avg output tokens/task':<25s}  {_fmt_tokens(avg_out_a):>{w}s}  {_fmt_tokens(avg_out_b):>{w}s}")

    # ── Per-repo breakdown ──
    if resolved_a or resolved_b:
        print()
        print("── Per-repo Resolve Rate ──")
        repo_ids: dict[str, list[str]] = defaultdict(list)
        for iid in all_ids:
            repo_ids[_repo_from_id(iid)].append(iid)

        print(f"  {'Repo':<30s}  {'N':>4s}  {label_a:>{w}s}  {label_b:>{w}s}")
        print(f"  {'─' * 30}  {'─' * 4}  {'─' * w}  {'─' * w}")

        for repo in sorted(repo_ids):
            ids = repo_ids[repo]
            n = len(ids)
            ra_repo = sum(1 for iid in ids if iid in resolved_a)
            rb_repo = sum(1 for iid in ids if iid in resolved_b)
            print(f"  {repo:<30s}  {n:>4d}  {_fmt_pct(ra_repo, n):>{w}s}  {_fmt_pct(rb_repo, n):>{w}s}")

        print(f"  {'─' * 30}  {'─' * 4}  {'─' * w}  {'─' * w}")
        total_n = len(all_ids)
        print(f"  {'TOTAL':<30s}  {total_n:>4d}  "
              f"{_fmt_pct(len(resolved_a), total_n):>{w}s}  "
              f"{_fmt_pct(len(resolved_b), total_n):>{w}s}")

    # ── Per-instance detail ──
    print()
    print("── Per-instance Detail ──")
    print(f"  {'Instance ID':<45s}  {'A':^6s}  {'B':^6s}  {'A time':>7s}  {'B time':>7s}")
    print(f"  {'─' * 45}  {'─' * 6}  {'─' * 6}  {'─' * 7}  {'─' * 7}")

    for iid in all_ids:
        ra = "PASS" if iid in resolved_a else ("FAIL" if iid in preds_a else "---")
        rb = "PASS" if iid in resolved_b else ("FAIL" if iid in preds_b else "---")
        ta = timing_a.get(iid, {}).get("agent_seconds", 0)
        tb = timing_b.get(iid, {}).get("agent_seconds", 0)
        print(f"  {iid:<45s}  {ra:^6s}  {rb:^6s}  {ta:>6.0f}s  {tb:>6.0f}s")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two SWE-bench agent runs")
    parser.add_argument("--a", type=Path, required=True, help="Results dir for agent A")
    parser.add_argument("--b", type=Path, required=True, help="Results dir for agent B")
    parser.add_argument("--label-a", default="Agent A", help="Display label for A")
    parser.add_argument("--label-b", default="Agent B", help="Display label for B")
    args = parser.parse_args()

    if not args.a.exists():
        print(f"ERROR: Results dir not found: {args.a}", file=sys.stderr)
        sys.exit(1)
    if not args.b.exists():
        print(f"ERROR: Results dir not found: {args.b}", file=sys.stderr)
        sys.exit(1)

    compare(args.a, args.b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
