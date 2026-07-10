#!/usr/bin/env python3
"""
Distill a teacher into a dataset. Default teacher: m.ai's own best model
served by mai_serve.py (self-distillation — we make our own).

Generates prompts across the potato curriculum (python, reasoning,
explanations, short stories — matching the TinyStories + python-codes
diet the earlier potatoes ate), asks the teacher model each one, and
writes {"prompt": ..., "response": ...} lines to a JSONL file.

Resume-safe: if the output file already exists, finished prompts are
skipped, so you can run it in bursts as the free tier allows.

Usage:
    python mai_serve.py &
    python train/distill_dataset.py --samples 2000 --out data/distill.jsonl
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mythos"))
from persona import DISTILL_SYSTEM_PROMPT
import mythos_client

# ── the potato curriculum ──────────────────────────────────────────────
TOPICS = {
    "python": [
        "Write a Python function that {task}. Include a short docstring.",
        "Show idiomatic Python to {task}, then explain it in two sentences.",
        "Write a small Python class that {task}.",
    ],
    "reasoning": [
        "Solve step by step: {task}",
        "Explain the reasoning, then give the final answer: {task}",
    ],
    "explain": [
        "Explain {task} simply, in under 150 words.",
        "Explain {task} to a beginner programmer with one concrete example.",
    ],
    "story": [
        "Write a very short story (under 120 words) about {task}.",
    ],
}

TASKS = {
    "python": [
        "reverses a linked list", "parses a CSV file into dictionaries",
        "implements binary search", "flattens a nested list",
        "checks if a string is a palindrome", "computes fibonacci with memoization",
        "merges two sorted lists", "counts word frequencies in a text file",
        "implements a simple LRU cache", "validates an email address with regex",
        "finds the longest common prefix of a list of strings",
        "implements bubble sort and explains its complexity",
        "reads JSON from a URL and handles errors", "implements a stack with a list",
        "generates all permutations of a string", "detects a cycle in a linked list",
        "implements a basic calculator for + - * /", "removes duplicates preserving order",
        "converts roman numerals to integers", "implements breadth-first search on a graph",
    ],
    "reasoning": [
        "If a train travels 120 km in 1.5 hours, what is its average speed?",
        "A bat and a ball cost $1.10 together; the bat costs $1 more than the ball. What does the ball cost?",
        "I have twice as many apples as oranges and 12 fruits total. How many oranges?",
        "What day of the week is 45 days after a Tuesday?",
        "If 3 workers build a wall in 6 hours, how long do 9 workers take?",
        "A number doubled and increased by 9 gives 25. What is it?",
        "How many times do the hands of a clock overlap in 12 hours?",
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
    ],
    "explain": [
        "recursion", "big-O notation", "how HTTP works", "what a hash map is",
        "the difference between processes and threads", "how gradient descent works",
        "what an API is", "how git branching works", "what a database index does",
        "how public-key cryptography works", "what Docker containers are",
        "the difference between TCP and UDP",
    ],
    "story": [
        "a rock that learns to code", "a potato that dreams of being a supercomputer",
        "a tiny robot's first day at school", "a kid who trains an AI in a basement",
        "a computer that falls asleep during training", "the last bug in the codebase",
    ],
}


def build_prompts(n, seed=1337):
    """n unique prompts. The base pool is every (template, task) combo;
    when that runs out we start variation rounds — same questions, asked
    again with a different-take marker, so the teacher (temp 0.7) gives
    fresh answers and every prompt string stays unique for resume."""
    rng = random.Random(seed)
    base = [
        template.format(task=task)
        for cat, templates in TOPICS.items()
        for template, task in itertools.product(templates, TASKS[cat])
    ]
    rng.shuffle(base)
    prompts = list(base)
    round_no = 2
    while len(prompts) < n:
        prompts.extend(f"{p}\n\n(Take {round_no}: give a different valid answer or approach.)"
                       for p in base)
        round_no += 1
    return prompts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", default="data/mythos_distill.jsonl")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between calls (rate limits)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[distill] resuming — {len(done)} samples already collected")

    prompts = [p for p in build_prompts(args.samples) if p not in done]
    print(f"[distill] {len(prompts)} prompts to collect from engine "
          f"'{mythos_client.ENGINE}' — the mountain is being milked 🏔️🥛")

    with open(args.out, "a") as out:
        for i, prompt in enumerate(prompts, 1):
            try:
                response = mythos_client.chat(
                    [{"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
                    temperature=0.7,
                )
            except SystemExit as e:
                print(f"[distill] stopping at {i - 1}/{len(prompts)}: {e}")
                break
            out.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"[distill] {i}/{len(prompts)} collected")
            time.sleep(args.sleep)

    print(f"[distill] done. Dataset at {args.out} — feed it to prepare_nanogpt_data.py")


if __name__ == "__main__":
    main()
