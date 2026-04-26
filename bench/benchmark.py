#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["regex", "tiktoken"]
# ///

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from minbpe import RegexTokenizer

ENCODE_DECODE_RUNS = 5
TRAINING_RUNS = 3
TRAINING_VOCAB_SIZES = [256 + 64, 512]
FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "taylorswift.txt"
RESULTS_DIR = Path(__file__).parent / "results"

SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def bench(label, runs, fn):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mn = min(times)
    mx = max(times)
    mean = sum(times) / len(times)
    print(f"  {label:<30}  min={mn:6.3f}s  mean={mean:6.3f}s  max={mx:6.3f}s")
    return {"label": label, "runs": runs, "times_s": times, "min_s": mn, "mean_s": mean, "max_s": mx}


fixture = FIXTURE_PATH.read_text(encoding="utf-8")

default_vocab_size = TRAINING_VOCAB_SIZES[0]
print(f"Training RegexTokenizer for encode/decode fixtures (vocab_size={default_vocab_size})...")
trained = RegexTokenizer()
trained.train(fixture, default_vocab_size)
trained.register_special_tokens(SPECIAL_TOKENS)
print("Done.\n")

plain_text = fixture * 3
parts = []
chars = list(fixture)
for i in range(0, len(chars), 500):
    parts.append("".join(chars[i:i + 500]) + "<|endoftext|>")
special_text = "".join(parts) * 3

plain_ids = trained.encode(plain_text)
special_ids = trained.encode(special_text, allowed_special="all")

print("Inputs:")
print(f"  plain:        {len(plain_text.encode())} bytes / {len(plain_ids)} tokens")
print(f"  with special: {len(special_text.encode())} bytes / {len(special_ids)} tokens")
print()

results = []

print(f"=== Encode / Decode ({ENCODE_DECODE_RUNS} runs) ===")
results.append(bench("encode:plain",          ENCODE_DECODE_RUNS, lambda: trained.encode(plain_text)))
results.append(bench("encode:special_tokens", ENCODE_DECODE_RUNS, lambda: trained.encode(special_text, allowed_special="all")))
results.append(bench("decode:plain",          ENCODE_DECODE_RUNS, lambda: trained.decode(plain_ids)))
results.append(bench("decode:special_tokens", ENCODE_DECODE_RUNS, lambda: trained.decode(special_ids)))

print(f"\n=== Training ({TRAINING_RUNS} runs) ===")
for vocab_size in TRAINING_VOCAB_SIZES:
    results.append(bench(f"train:{vocab_size}", TRAINING_RUNS, lambda vs=vocab_size: RegexTokenizer().train(fixture, vs)))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
try:
    git_sha = subprocess.check_output(
        ["git", "-C", str(Path(__file__).parent.parent), "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    git_sha = None

output = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
        "platform": sys.platform,
        "git_sha": git_sha or None,
        "fixture_bytes": len(fixture.encode()),
        "plain_text_bytes": len(plain_text.encode()),
        "plain_token_count": len(plain_ids),
        "special_text_bytes": len(special_text.encode()),
        "special_token_count": len(special_ids),
    },
    "results": results,
}

output_path = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
output_path.write_text(json.dumps(output, indent=2))
print(f"\nSaved: {output_path}")
