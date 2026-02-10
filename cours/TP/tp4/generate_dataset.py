"""
GSM8K DASD Two-Stage Distillation Pipeline
===========================================
Two-stage knowledge distillation with structured reasoning:
  - Stage 1 (low τ ≈ 0.3): deterministic, high-confidence answers
  - Stage 2 (high τ ≈ 0.9): diverse, exploratory answers
A single probability (0-1) is stored per response for downstream DASD filtering.
Results saved as JSON, CSV, HuggingFace Dataset, and Llama-Factory ShareGPT format.

Usage:
    python generate_dataset.py -n 100 -d 1.5
    python generate_dataset.py -n 500 --stage1-temp 0.15 --stage2-temp 0.95
    python generate_dataset.py -n 50 --max-retries 5 --min-probability 0.3
"""

import argparse
import csv
import json
import math
import os
import random
import re
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openai
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
DEFAULT_MODEL = "qwen3"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_DELAY = 1.0
DEFAULT_MAX_TOKENS = 5000
DEFAULT_TOP_LOGPROBS = 1
DEFAULT_STAGE1_TEMP = 0.3
DEFAULT_STAGE2_TEMP = 0.9
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MIN_PROBABILITY = 0.0  # minimum probability (0-1) to keep a response
DEFAULT_MIN_LENGTH = 20  # minimum character length for a valid response
LOCAL_CACHE_DIR = "./data/gsm8k"

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "showing your work clearly. End with the final numerical answer "
    "on its own line, formatted as: #### <number>"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """Result of a single-stage model query."""
    stage: int
    temperature: float
    answer: str
    probability: float | None  # average token probability (0-1)
    final_answer: str  # numerical answer extracted after ####
    passed_quality: bool


@dataclass
class SampleRecord:
    """Full record for one dataset sample across both stages."""
    index: int
    question: str
    original_answer: str
    stage1: dict[str, Any] | None = None
    stage2: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def download_gsm8k(cache_dir: str = LOCAL_CACHE_DIR):
    """Download GSM8K from HuggingFace and cache locally."""
    print("Downloading GSM8K from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main")
    ds.save_to_disk(cache_dir)
    print(f"Dataset cached at {cache_dir}")
    return ds


def load_gsm8k(cache_dir: str = LOCAL_CACHE_DIR):
    """Load GSM8K from local cache, downloading if necessary."""
    if Path(cache_dir).exists():
        print(f"Loading GSM8K from local cache ({cache_dir})...")
        return load_from_disk(cache_dir)
    return download_gsm8k(cache_dir)


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------
def build_client(base_url: str, api_key: str) -> openai.OpenAI:
    """Create an OpenAI-compatible client."""
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def query_model_with_retry(
    client: openai.OpenAI,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
):
    """
    Query the model with exponential-backoff retry on transient errors.
    Returns the raw API response or raises after exhausting retries.
    """
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            return response

        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            last_exc = exc
            wait = retry_backoff ** attempt
            print(f"    ⚠ Attempt {attempt}/{max_retries} failed ({type(exc).__name__}), "
                  f"retrying in {wait:.1f}s...")
            time.sleep(wait)

        except openai.APIError as exc:
            # Non-retryable API errors
            raise exc

    raise RuntimeError(
        f"All {max_retries} attempts failed. Last error: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------
def extract_probability(response) -> float | None:
    """
    Compute a single average token probability (0-1) from the API response.
    Returns None if logprobs are unavailable.
    """
    logprobs_data = response.choices[0].logprobs
    if not logprobs_data or not logprobs_data.content:
        return None

    avg_logprob = sum(tok.logprob for tok in logprobs_data.content) / len(logprobs_data.content)
    return math.exp(avg_logprob)


# ---------------------------------------------------------------------------
# Response parsing & quality filtering
# ---------------------------------------------------------------------------
def extract_final_answer(text: str) -> str:
    """
    Extract the numerical answer from the response.
    Tries #### marker first (GSM8K format), then falls back to \\boxed{}.
    Returns an empty string if neither is found.
    """
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()

    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    return ""


def check_quality(
    answer: str,
    probability: float | None,
    final_answer: str,
    min_probability: float,
    min_length: int,
) -> bool:
    """
    Apply quality filters to a model response.
    Returns True if the response passes all checks.
    """
    if len(answer.strip()) < min_length:
        return False

    if probability is not None and probability < min_probability:
        return False

    if not final_answer:
        print("    ℹ Response missing #### marker (kept but flagged)")

    return True


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------
def run_stage(
    client: openai.OpenAI,
    model: str,
    question: str,
    stage: int,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
    max_retries: int,
    retry_backoff: float,
    min_probability: float,
    min_length: int,
) -> StageResult | None:
    """Run a single generation stage and return the parsed result."""
    try:
        response = query_model_with_retry(
            client, model, question,
            temperature=temperature,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
    except Exception as exc:
        print(f"    ✗ Stage {stage} failed: {exc}")
        return None

    raw_answer = response.choices[0].message.content or ""
    probability = extract_probability(response)
    final_answer = extract_final_answer(raw_answer)
    passed = check_quality(raw_answer, probability, final_answer, min_probability, min_length)

    prob_str = f"{probability:.4f}" if probability is not None else "N/A"
    status = "✓" if passed else "✗ (filtered)"
    print(f"    Stage {stage} (τ={temperature}) — probability: {prob_str} {status}")

    return StageResult(
        stage=stage,
        temperature=temperature,
        answer=raw_answer,
        probability=probability,
        final_answer=final_answer,
        passed_quality=passed,
    )


# ---------------------------------------------------------------------------
# Output: JSON, CSV, HuggingFace, Llama-Factory
# ---------------------------------------------------------------------------
def generate_run_id(length: int = 6) -> str:
    """Generate a random alphanumeric run identifier."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def stage_to_dict(result: StageResult | None) -> dict[str, Any] | None:
    """Convert a StageResult to a serializable dict."""
    if result is None:
        return None
    return asdict(result)


def save_json(records: list[dict], path: Path):
    """Save full records as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved to {path}")


def save_csv(records: list[dict], path: Path):
    """Save a flat CSV summary."""
    rows = []
    for r in records:
        base = {
            "index": r["index"],
            "question": r["question"],
            "original_answer": r["original_answer"],
        }
        for stage_key in ("stage1", "stage2"):
            s = r.get(stage_key)
            if s:
                base[f"{stage_key}_answer"] = s["answer"]
                base[f"{stage_key}_final_answer"] = s["final_answer"]
                base[f"{stage_key}_probability"] = s["probability"]
                base[f"{stage_key}_temperature"] = s["temperature"]
                base[f"{stage_key}_passed_quality"] = s["passed_quality"]
            else:
                base[f"{stage_key}_answer"] = ""
                base[f"{stage_key}_final_answer"] = ""
                base[f"{stage_key}_probability"] = None
                base[f"{stage_key}_temperature"] = None
                base[f"{stage_key}_passed_quality"] = False
        rows.append(base)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved to {path}")


def save_hf_dataset(records: list[dict], path: Path):
    """Save a HuggingFace Dataset (Arrow format), flattened for tabular storage."""
    flat = []
    for r in records:
        row = {
            "index": r["index"],
            "question": r["question"],
            "original_answer": r["original_answer"],
        }
        for stage_key in ("stage1", "stage2"):
            s = r.get(stage_key)
            row[f"{stage_key}_answer"] = s["answer"] if s else ""
            row[f"{stage_key}_final_answer"] = s["final_answer"] if s else ""
            row[f"{stage_key}_probability"] = s["probability"] if s else None
            row[f"{stage_key}_temperature"] = s["temperature"] if s else None
            row[f"{stage_key}_passed_quality"] = s["passed_quality"] if s else False
        flat.append(row)

    ds = Dataset.from_list(flat)
    ds.save_to_disk(str(path))
    print(f"  HuggingFace dataset saved to {path}")


# ---------------------------------------------------------------------------
# Llama-Factory format (ShareGPT)
# ---------------------------------------------------------------------------
def convert_to_sharegpt(records: list[dict], stage_key: str = "stage1") -> list[dict]:
    """
    Convert records to ShareGPT format for Llama-Factory.
    Only includes samples that passed quality filtering.
    """
    sharegpt = []
    for r in records:
        s = r.get(stage_key)
        if not s or not s["passed_quality"]:
            continue

        sharegpt.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": r["question"]},
                {"from": "gpt", "value": s["answer"]},
            ]
        })
    return sharegpt


def save_sharegpt(records: list[dict], output_dir: Path):
    """Save ShareGPT files for each stage and a combined version."""
    lf_dir = output_dir / "llama_factory"
    lf_dir.mkdir(parents=True, exist_ok=True)

    for stage_key, label in [("stage1", "low_temp"), ("stage2", "high_temp")]:
        sharegpt = convert_to_sharegpt(records, stage_key)
        if sharegpt:
            path = lf_dir / f"sharegpt_{label}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sharegpt, f, ensure_ascii=False, indent=2)
            print(f"  ShareGPT ({label}): {len(sharegpt)} samples → {path}")

    # Combined (both stages merged, deduplicated by preferring stage1)
    combined = convert_to_sharegpt(records, "stage1")
    stage1_questions = {r["question"] for r in records if r.get("stage1") and r["stage1"]["passed_quality"]}
    for r in records:
        s2 = r.get("stage2")
        if s2 and s2["passed_quality"] and r["question"] not in stage1_questions:
            combined.append({
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": r["question"]},
                    {"from": "gpt", "value": s2["answer"]},
                ]
            })

    if combined:
        path = lf_dir / "sharegpt_combined.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"  ShareGPT (combined): {len(combined)} samples → {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace):
    """Execute the full two-stage DASD generation pipeline."""

    # --- Resolve API key ---
    api_key = args.api_key or os.getenv("INFOMANIAK_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: No API key. Use --api-key or set INFOMANIAK_API_KEY.")

    # --- Load source dataset ---
    ds = load_gsm8k(args.cache_dir)
    train_split = ds["train"]
    num_samples = min(args.num_samples, len(train_split))

    print(f"\n{'=' * 60}")
    print(f"  DASD Two-Stage Distillation Pipeline")
    print(f"  Model:       {args.model}")
    print(f"  Samples:     {num_samples} / {len(train_split)}")
    print(f"  Stage 1 τ:   {args.stage1_temp}")
    print(f"  Stage 2 τ:   {args.stage2_temp}")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Delay:       {args.delay}s")
    print(f"  Min prob:    {args.min_probability}")
    print(f"{'=' * 60}\n")

    # --- Build client ---
    client = build_client(args.base_url, api_key)

    # --- Query loop ---
    records: list[dict] = []
    stats = {"stage1_pass": 0, "stage2_pass": 0, "stage1_fail": 0, "stage2_fail": 0}

    for i in range(num_samples):
        question = train_split[i]["question"]
        original_answer = train_split[i]["answer"]

        print(f"[{i + 1}/{num_samples}] {question[:80]}...")

        # Stage 1: low temperature (deterministic)
        s1 = run_stage(
            client, args.model, question,
            stage=1, temperature=args.stage1_temp,
            max_tokens=args.max_tokens, top_logprobs=args.top_logprobs,
            max_retries=args.max_retries, retry_backoff=args.retry_backoff,
            min_probability=args.min_probability, min_length=DEFAULT_MIN_LENGTH,
        )

        if args.delay > 0:
            time.sleep(args.delay)

        # Stage 2: high temperature (diverse)
        s2 = run_stage(
            client, args.model, question,
            stage=2, temperature=args.stage2_temp,
            max_tokens=args.max_tokens, top_logprobs=args.top_logprobs,
            max_retries=args.max_retries, retry_backoff=args.retry_backoff,
            min_probability=args.min_probability, min_length=DEFAULT_MIN_LENGTH,
        )

        # Track stats
        if s1 and s1.passed_quality:
            stats["stage1_pass"] += 1
        else:
            stats["stage1_fail"] += 1
        if s2 and s2.passed_quality:
            stats["stage2_pass"] += 1
        else:
            stats["stage2_fail"] += 1

        records.append({
            "index": i,
            "question": question,
            "original_answer": original_answer,
            "stage1": stage_to_dict(s1),
            "stage2": stage_to_dict(s2),
        })

        # Delay between samples (skip after last)
        if i < num_samples - 1 and args.delay > 0:
            time.sleep(args.delay)

    # --- Save all outputs ---
    run_id = generate_run_id()
    output_dir = Path(args.output_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Saving results to {output_dir}")
    print(f"{'=' * 60}")

    save_json(records, output_dir / "results.json")
    save_csv(records, output_dir / "results.csv")
    save_hf_dataset(records, output_dir / "hf_dataset")
    save_sharegpt(records, output_dir)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete")
    print(f"  Total samples:     {len(records)}")
    print(f"  Stage 1 passed:    {stats['stage1_pass']} / {len(records)}")
    print(f"  Stage 2 passed:    {stats['stage2_pass']} / {len(records)}")
    print(f"  Output directory:  {output_dir}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DASD two-stage distillation pipeline for GSM8K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    p.add_argument("-n", "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="Number of samples to process")
    p.add_argument("--cache-dir", type=str, default=LOCAL_CACHE_DIR,
                   help="Local cache directory for GSM8K")

    # Model
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                   help="Teacher model identifier")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible API base URL")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key (overrides INFOMANIAK_API_KEY env var)")

    # Generation
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="Max tokens per response")
    p.add_argument("--top-logprobs", type=int, default=DEFAULT_TOP_LOGPROBS,
                   help="Number of top logprobs per token")
    p.add_argument("--stage1-temp", type=float, default=DEFAULT_STAGE1_TEMP,
                   help="Stage 1 temperature (low, deterministic)")
    p.add_argument("--stage2-temp", type=float, default=DEFAULT_STAGE2_TEMP,
                   help="Stage 2 temperature (high, diverse)")

    # Retry & rate-limiting
    p.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY,
                   help="Delay in seconds between API calls")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help="Max retries on transient API errors")
    p.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF,
                   help="Exponential backoff base for retries")

    # Quality filtering
    p.add_argument("--min-probability", type=float, default=DEFAULT_MIN_PROBABILITY,
                   help="Minimum probability (0-1) to keep a response")
    p.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH,
                   help="Minimum character length for a valid response")

    # Output
    p.add_argument("-o", "--output-dir", type=str, default="./outputs",
                   help="Root directory for generated datasets")

    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())