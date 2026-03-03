#!/usr/bin/env python3
"""
Benchmark request-level static batching vs continuous slot reuse on vanilla HF.

This script intentionally uses Hugging Face `transformers` model forward calls
without vLLM runtime features, to isolate scheduler policy effects.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RequestSpec:
    request_id: str
    prompt_len: int
    target_new_tokens: int


class RequestState:
    def __init__(self, spec: RequestSpec, prompt_ids: torch.Tensor):
        self.spec = spec
        self.input_ids = prompt_ids
        self.remaining = spec.target_new_tokens
        self.generated = 0
        self.first_token_at_s: float | None = None
        self.finished_at_s: float | None = None

    def to_record(self, start_s: float) -> dict:
        ttft_ms = None
        e2el_ms = None
        if self.first_token_at_s is not None:
            ttft_ms = (self.first_token_at_s - start_s) * 1000.0
        if self.finished_at_s is not None:
            e2el_ms = (self.finished_at_s - start_s) * 1000.0
        return {
            "request_id": self.spec.request_id,
            "prompt_len": self.spec.prompt_len,
            "target_new_tokens": self.spec.target_new_tokens,
            "generated_new_tokens": self.generated,
            "ttft_ms": ttft_ms,
            "e2el_ms": e2el_ms,
        }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[lower]
    w_upper = k - lower
    w_lower = 1.0 - w_upper
    return ordered[lower] * w_lower + ordered[upper] * w_upper


def now_s(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def greedy_step(
    model: AutoModelForCausalLM,
    sequences: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(sequences)
    max_len = max(seq.numel() for seq in sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(sequences):
        length = seq.numel()
        input_ids[idx, :length] = seq
        attention_mask[idx, :length] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return torch.argmax(logits[:, -1, :], dim=-1)


def make_request_specs(num_requests: int) -> list[RequestSpec]:
    # Mix short/medium/long decode targets to surface static-batch inefficiency.
    prompt_pattern = [128, 256, 384, 512]
    decode_pattern = [24, 24, 24, 96, 24, 24, 24, 128]

    specs: list[RequestSpec] = []
    for i in range(num_requests):
        specs.append(
            RequestSpec(
                request_id=f"req_{i:03d}",
                prompt_len=prompt_pattern[i % len(prompt_pattern)],
                target_new_tokens=decode_pattern[i % len(decode_pattern)],
            )
        )
    return specs


def build_prompt_ids(spec: RequestSpec, bos_token_id: int, vocab_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    body_len = max(spec.prompt_len - 1, 0)
    if body_len == 0:
        return torch.tensor([bos_token_id], dtype=torch.long)
    body = torch.randint(
        low=10,
        high=max(11, vocab_size - 1),
        size=(body_len,),
        dtype=torch.long,
        generator=generator,
    )
    return torch.cat([torch.tensor([bos_token_id], dtype=torch.long), body], dim=0)


def make_states(
    specs: list[RequestSpec],
    bos_token_id: int,
    vocab_size: int,
    device: torch.device,
) -> list[RequestState]:
    states: list[RequestState] = []
    for idx, spec in enumerate(specs):
        prompt = build_prompt_ids(spec, bos_token_id=bos_token_id, vocab_size=vocab_size, seed=1000 + idx)
        states.append(RequestState(spec=spec, prompt_ids=prompt.to(device)))
    return states


def summarize(
    mode: str,
    states: list[RequestState],
    start_s: float,
    end_s: float,
    wasted_decode_slots: int,
) -> dict:
    wall_s = end_s - start_s
    generated_tokens = sum(state.generated for state in states)
    ttfts = [(state.first_token_at_s - start_s) * 1000.0 for state in states if state.first_token_at_s is not None]
    e2els = [(state.finished_at_s - start_s) * 1000.0 for state in states if state.finished_at_s is not None]
    return {
        "mode": mode,
        "num_requests": len(states),
        "total_generated_tokens": generated_tokens,
        "wall_time_s": wall_s,
        "request_throughput_req_s": len(states) / wall_s if wall_s > 0 else float("nan"),
        "output_token_throughput_tok_s": generated_tokens / wall_s if wall_s > 0 else float("nan"),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p99_ms": percentile(ttfts, 99),
        "e2el_p50_ms": percentile(e2els, 50),
        "e2el_p99_ms": percentile(e2els, 99),
        "wasted_decode_slots": wasted_decode_slots,
        "requests": [state.to_record(start_s) for state in states],
    }


def run_request_level_static(
    model: AutoModelForCausalLM,
    states: list[RequestState],
    batch_size: int,
    pad_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> dict:
    start_s = now_s(device)
    wasted_decode_slots = 0
    eos_tensor = torch.tensor([eos_token_id], dtype=torch.long, device=device)

    for offset in range(0, len(states), batch_size):
        batch = states[offset : offset + batch_size]
        max_steps = max(state.remaining for state in batch)
        for _ in range(max_steps):
            next_tokens = greedy_step(model, [state.input_ids for state in batch], pad_token_id=pad_token_id, device=device)
            step_time_s = now_s(device)
            done_in_batch = 0
            for idx, state in enumerate(batch):
                if state.remaining > 0:
                    token_tensor = next_tokens[idx : idx + 1]
                    state.input_ids = torch.cat([state.input_ids, token_tensor], dim=0)
                    state.generated += 1
                    state.remaining -= 1
                    if state.first_token_at_s is None:
                        state.first_token_at_s = step_time_s
                    if state.remaining == 0:
                        state.finished_at_s = step_time_s
                else:
                    # Keep done requests in the static batch to model scheduler waste.
                    state.input_ids = torch.cat([state.input_ids, eos_tensor], dim=0)
                    done_in_batch += 1
            wasted_decode_slots += done_in_batch

    end_s = now_s(device)
    return summarize(
        mode="request_level_static",
        states=states,
        start_s=start_s,
        end_s=end_s,
        wasted_decode_slots=wasted_decode_slots,
    )


def run_continuous_slot_reuse(
    model: AutoModelForCausalLM,
    states: list[RequestState],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict:
    start_s = now_s(device)
    waiting = list(states)
    active: list[RequestState] = []

    while waiting and len(active) < batch_size:
        active.append(waiting.pop(0))

    while active:
        next_tokens = greedy_step(model, [state.input_ids for state in active], pad_token_id=pad_token_id, device=device)
        step_time_s = now_s(device)

        finished: list[int] = []
        for idx, state in enumerate(active):
            token_tensor = next_tokens[idx : idx + 1]
            state.input_ids = torch.cat([state.input_ids, token_tensor], dim=0)
            state.generated += 1
            state.remaining -= 1
            if state.first_token_at_s is None:
                state.first_token_at_s = step_time_s
            if state.remaining == 0:
                state.finished_at_s = step_time_s
                finished.append(idx)

        for idx in reversed(finished):
            del active[idx]
        while waiting and len(active) < batch_size:
            active.append(waiting.pop(0))

    end_s = now_s(device)
    return summarize(
        mode="continuous_slot_reuse",
        states=states,
        start_s=start_s,
        end_s=end_s,
        wasted_decode_slots=0,
    )


def format_brief_table(result: dict) -> str:
    return (
        f"{result['mode']:>21} | "
        f"{result['request_throughput_req_s']:.4f} req/s | "
        f"{result['output_token_throughput_tok_s']:.2f} tok/s | "
        f"TTFT p50 {result['ttft_p50_ms']:.2f} ms | "
        f"E2EL p50 {result['e2el_p50_ms']:.2f} ms | "
        f"wasted slots {result['wasted_decode_slots']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HF scheduler benchmark: request-level vs continuous")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_token_id

    print(f"[info] loading model={args.model} device={device} dtype={dtype}", flush=True)
    load_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        # Reduce peak memory by avoiding a full CPU->GPU duplicate copy during .to(device).
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device.type != "cuda":
        model.to(device)
    print("[info] model loaded", flush=True)
    model.eval()

    specs = make_request_specs(num_requests=args.num_requests)
    vocab_size = int(model.config.vocab_size)

    static_states = make_states(specs, bos_token_id=bos_token_id, vocab_size=vocab_size, device=device)
    continuous_states = make_states(specs, bos_token_id=bos_token_id, vocab_size=vocab_size, device=device)

    print("[info] running request_level_static", flush=True)
    static_result = run_request_level_static(
        model=model,
        states=static_states,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        device=device,
    )
    print("[info] running continuous_slot_reuse", flush=True)
    continuous_result = run_continuous_slot_reuse(
        model=model,
        states=continuous_states,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
        device=device,
    )

    report = {
        "meta": {
            "model": args.model,
            "device": str(device),
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "num_requests": args.num_requests,
            "prompt_len_pattern": [128, 256, 384, 512],
            "target_new_tokens_pattern": [24, 24, 24, 96, 24, 24, 24, 128],
            "note": (
                "All requests are treated as arriving at t=0. "
                "The static mode keeps finished requests in the batch until the longest request ends."
            ),
        },
        "results": {
            "request_level_static": static_result,
            "continuous_slot_reuse": continuous_result,
        },
        "summary_delta": {
            "request_throughput_ratio_cont_over_static": (
                continuous_result["request_throughput_req_s"] / static_result["request_throughput_req_s"]
            ),
            "output_throughput_ratio_cont_over_static": (
                continuous_result["output_token_throughput_tok_s"] / static_result["output_token_throughput_tok_s"]
            ),
            "ttft_p50_ratio_static_over_cont": static_result["ttft_p50_ms"] / continuous_result["ttft_p50_ms"],
            "e2el_p50_ratio_static_over_cont": static_result["e2el_p50_ms"] / continuous_result["e2el_p50_ms"],
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(format_brief_table(static_result))
    print(format_brief_table(continuous_result))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
