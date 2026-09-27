import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.shard < args.shards:
        raise ValueError("Benchmark needs a fresh output and a valid shard")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError("Each inference benchmark worker must own one allocated GPU")
    from vllm import LLM, SamplingParams

    data = json.loads(args.prompts.read_text())
    records = data[args.shard :: args.shards]
    if len(data) != 512 or not records:
        raise ValueError("Expected the complete 512-response production prompt workload")
    started = time.monotonic()
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        seed=42,
        max_model_len=10240,
        max_num_seqs=args.sequences,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        worker_extension_cls="prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
        additional_config={"fp32_lm_head": True, "fp32_router_logits": True},
    )
    initialized = time.monotonic()
    engine.generate(
        [{"prompt_token_ids": records[0]["prompt_token_ids"]}],
        SamplingParams(temperature=1.0, max_tokens=32, seed=42),
        use_tqdm=False,
    )
    warm = time.monotonic()
    prompts = [{"prompt_token_ids": record["prompt_token_ids"]} for record in records]
    parameters = [
        SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=8192, logprobs=0, seed=record["seed"])
        for record in records
    ]
    outputs = engine.generate(prompts, parameters, use_tqdm=False)
    elapsed = time.monotonic() - warm
    if len(outputs) != len(records):
        raise RuntimeError("The benchmark dropped requests")
    responses = []
    for record, output in zip(records, outputs, strict=True):
        if output.prompt_token_ids != record["prompt_token_ids"] or len(output.outputs) != 1:
            raise RuntimeError("The benchmark changed prompt identity or response multiplicity")
        result = output.outputs[0]
        if not result.token_ids or len(result.token_ids) > 8192 or result.finish_reason not in {"stop", "length"}:
            raise RuntimeError("The benchmark produced an invalid completion")
        if result.logprobs is None or len(result.logprobs) != len(result.token_ids):
            raise RuntimeError("Sampling log-probabilities are missing")
        logprobs = [row[token].logprob for token, row in zip(result.token_ids, result.logprobs, strict=True)]
        if not all(math.isfinite(value) for value in logprobs):
            raise FloatingPointError("The benchmark produced nonfinite sampling log-probabilities")
        responses.append(
            {
                "index": record["index"],
                "seed": record["seed"],
                "tokens": len(result.token_ids),
                "finish_reason": result.finish_reason,
                "token_sha256": hashlib.sha256(json.dumps(result.token_ids).encode()).hexdigest(),
                "logprob_sum": sum(logprobs),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "sequences": args.sequences,
                "shard": args.shard,
                "shards": args.shards,
                "device": os.environ["CUDA_VISIBLE_DEVICES"],
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "PYTORCH_CUDA_ALLOC_CONF",
                        "VLLM_WORKER_MULTIPROC_METHOD",
                        "OMP_NUM_THREADS",
                        "NCCL_P2P_DISABLE",
                        "NCCL_SHM_DISABLE",
                    )
                },
                "initialization_seconds": initialized - started,
                "warmup_seconds": warm - initialized,
                "generation_seconds": elapsed,
                "output_tokens": sum(row["tokens"] for row in responses),
                "tokens_per_second": sum(row["tokens"] for row in responses) / elapsed,
                "responses": responses,
                "scope": "isolated single-engine batching; no production router, grading, or weight transfer",
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
