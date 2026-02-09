"""
GSM8K Dataset Re-generation Pipeline
=====================================
Downloads GSM8K, queries a teacher model for new answers,
computes confidence scores, and saves the augmented dataset
in both HuggingFace and CSV formats.
"""

import argparse
import csv
import math
import os
import random
import string
import time
from pathlib import Path

import openai
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
DEFAULT_MODEL = "qwen3"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_DELAY = 1.0  # seconds between API calls
DEFAULT_MAX_TOKENS = 5000
DEFAULT_TEMPERATURE = 0.15
LOCAL_CACHE_DIR = "./data/gsm8k"


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


def query_model(
    client: openai.OpenAI,
    model: str,
    question: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """Send a question to the teacher model and return the raw response."""
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
    )


def calculate_confidence(response) -> float | None:
    """
    Compute average token-level confidence (%) from logprobs.
    Returns None when logprobs are unavailable.
    """
    logprobs_data = response.choices[0].logprobs
    if not logprobs_data or not logprobs_data.content:
        return None

    total_logprob = sum(tok.logprob for tok in logprobs_data.content)
    token_count = len(logprobs_data.content)
    avg_logprob = total_logprob / token_count
    return math.exp(avg_logprob) * 100


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def generate_run_id(length: int = 6) -> str:
    """Generate a random alphanumeric run identifier."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def save_results(records: list[dict], output_dir: str):
    """
    Save results as:
      - A HuggingFace Dataset (Arrow format)
      - A CSV file with confidence scores
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # HuggingFace dataset
    hf_dataset = Dataset.from_list(records)
    hf_path = out / "dataset"
    hf_dataset.save_to_disk(str(hf_path))
    print(f"HuggingFace dataset saved to {hf_path}")

    # CSV
    csv_path = out / "results.csv"
    fieldnames = ["index", "question", "original_answer", "model_answer", "confidence"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV saved to {csv_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace):
    """Execute the full generation pipeline."""

    # --- Resolve API key ---
    api_key = args.api_key or os.getenv("INFOMANIAK_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: No API key provided (use --api-key or set INFOMANIAK_API_KEY).")

    # --- Load source dataset ---
    ds = load_gsm8k(args.cache_dir)
    train_split = ds["train"]
    num_samples = min(args.num_samples, len(train_split))
    print(f"Processing {num_samples} / {len(train_split)} samples")

    # --- Build client ---
    client = build_client(args.base_url, api_key)

    # --- Query loop ---
    records: list[dict] = []

    for i in range(num_samples):
        question = train_split[i]["question"]
        original_answer = train_split[i]["answer"]

        print(f"\n[{i + 1}/{num_samples}] Querying model...")
        try:
            response = query_model(
                client,
                model=args.model,
                question=question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            model_answer = response.choices[0].message.content
            confidence = calculate_confidence(response)
        except Exception as exc:
            print(f"  ⚠ API error: {exc}")
            model_answer = ""
            confidence = None

        conf_str = f"{confidence:.2f}%" if confidence is not None else "N/A"
        print(f"  Confidence: {conf_str}")

        records.append(
            {
                "index": i,
                "question": question,
                "original_answer": original_answer,
                "model_answer": model_answer,
                "confidence": confidence,
            }
        )

        # Respect rate-limiting delay (skip after last item)
        if i < num_samples - 1 and args.delay > 0:
            time.sleep(args.delay)

    # --- Save outputs ---
    run_id = generate_run_id()
    output_dir = Path(args.output_dir) / f"run_{run_id}"
    save_results(records, str(output_dir))

    print(f"\n Done — {len(records)} samples saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-generate GSM8K answers with a teacher model and build a new dataset."
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples to process (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay in seconds between API calls (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Teacher model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (overrides INFOMANIAK_API_KEY env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens for model response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=LOCAL_CACHE_DIR,
        help=f"Local cache directory for GSM8K (default: {LOCAL_CACHE_DIR})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./outputs",
        help="Root directory for generated datasets (default: ./outputs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())