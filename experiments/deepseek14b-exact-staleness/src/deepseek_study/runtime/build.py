import json
from pathlib import Path

from deepseek_study.runtime.checkpoints import atomic_write
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import write_subconfigs


def resolve(study, resume=None):
    directory = study.output_dir.resolve()
    model = str(study.prepared_model_path.resolve())
    scheduler = {"type": "constant"}
    if study.lr_warmup_steps or study.lr_decay_steps:
        scheduler = {
            "type": "linear",
            "warmup_steps": study.lr_warmup_steps,
            "decay_steps": study.lr_decay_steps,
            "min_lr": study.min_learning_rate,
        }
    config = RLConfig.model_validate(
        {
            "model": {"name": model},
            "tokenizer": {"name": model, "trust_remote_code": False},
            "output_dir": str(directory.parent),
            "run": {"name": directory.name, "dir": directory.name},
            "seq_len": study.sequence_length,
            "max_steps": study.max_steps,
            "dashboard": False,
            "monitors": {"file": {}},
            "ckpt": {"interval": study.checkpoint_interval},
            "resume": {"dir": str(Path(resume).resolve())} if resume else None,
            "deployment": {
                "type": "single_node",
                "gpus_per_node": study.trainer_gpus + study.inference_gpus,
                "num_train_gpus": study.trainer_gpus,
                "num_infer_gpus": study.inference_gpus,
            },
            "weight_broadcast": {"type": "nccl", "port": study.inference_port + 10, "timeout": study.timeout_seconds},
            "rollout_transport": {"type": "zmq", "port": study.inference_port + 20},
            "trainer": {
                "model": {
                    "impl": "hf",
                    "attn": {"fa2": "flash_attention_2", "fa3": "flash_attention_3", "fa4": "flash_attention_4"}[
                        study.trainer_attention
                    ],
                    "compile": {} if study.trainer_compile else None,
                    "ac": {} if study.activation_checkpointing else None,
                    "ac_offloading": {} if study.activation_cpu_offload else None,
                    "fused_lm_head_token_chunk_size": study.lm_head_chunk_size,
                    "dp_replicate": 1,
                    "reshard_after_forward": True,
                    "optim_cpu_offload": study.optimizer_cpu_offload,
                    "fsdp_cpu_offload": False,
                    "optimization_dtype": "float32",
                    "reduce_dtype": "float32",
                    "quantization": None,
                    "ep": 1,
                    "cp": 1,
                },
                "loss": {
                    "type": "custom",
                    "import_path": "deepseek_study.learning.loss.clipped_grpo",
                    "kwargs": {"clip_epsilon": study.clip_epsilon},
                },
                "optim": {
                    "type": "adamw",
                    "lr": study.learning_rate,
                    "weight_decay": study.weight_decay,
                    "max_norm": study.max_grad_norm,
                    "betas1": study.adam_beta1,
                    "betas2": study.adam_beta2,
                },
                "scheduler": scheduler,
            },
            "orchestrator": {
                "eval": None,
                "model": {"client": {"base_url": f"http://127.0.0.1:{study.inference_port}/v1"}},
                "renderer": {"name": "default"},
                "batch_size": study.response_batch_size,
                "group_size": study.responses_per_prompt,
                "max_off_policy_steps": study.lag,
                "concurrency": {
                    "initial_inflight": study.rollout_concurrency,
                    "min_inflight": study.rollout_concurrency,
                    "max_inflight": study.rollout_concurrency,
                },
                "env_server_base_port": study.inference_port + 30,
                "train": {
                    "filter_zero_advantages": False,
                    "sampling": {
                        "temperature": study.temperature,
                        "max_completion_tokens": study.response_max_tokens,
                        "extra_body": {
                            "top_k": -1,
                            "min_p": 0.0,
                            "repetition_penalty": 1.0,
                            "presence_penalty": 0.0,
                            "frequency_penalty": 0.0,
                        },
                    },
                    "source": [
                        {
                            "name": "cleaned-deepscaler",
                            "env": {
                                "agent": {
                                    "runtime": {"type": "subprocess"},
                                    "harness": {"id": "null"},
                                    "retries": {"max_retries": 0},
                                },
                                "retries": {"max_retries": 0},
                                "taskset": {
                                    "id": "deepseek-deepscaler",
                                    "dataset_path": str(study.dataset_path.resolve()),
                                    "data_manifest": str(study.data_manifest.resolve()),
                                    "tokenizer_path": model,
                                    "prompt_instruction": study.prompt_instruction,
                                    "truncated_reward": study.truncated_reward,
                                    "reward_timeout_seconds": study.reward_timeout_seconds,
                                    "reward_outer_timeout_seconds": study.reward_outer_timeout_seconds,
                                    "reward_workers": study.reward_workers,
                                    "reward_retries": study.reward_retries,
                                    "seed": study.seed,
                                },
                            },
                        }
                    ],
                },
            },
            "inference": {
                "router": None,
                "server": {"host": "127.0.0.1", "port": study.inference_port},
                "vllm": {
                    "dtype": "bfloat16",
                    "tokenizer": model,
                    "max_model_len": study.sequence_length,
                    "max_num_seqs": study.inference_max_sequences,
                    "tensor_parallel_size": study.inference_tensor_parallel,
                    "data_parallel_size": study.inference_gpus // study.inference_tensor_parallel,
                    "gpu_memory_utilization": study.inference_memory_utilization,
                    "seed": study.seed,
                    "generation_config": "vllm",
                    "logprobs_mode": "raw_logprobs",
                    "enable_prefix_caching": False,
                    "tool_call_parser": None,
                    "reasoning_parser": None,
                    "quantization": None,
                },
            },
        }
    )
    if config.trainer.output_dir.resolve() != directory or config.orchestrator.output_dir.resolve() != directory:
        raise ValueError("Upstream run directory resolution changed")
    return config


def build(study, destination, resume=None):
    destination = Path(destination)
    config = resolve(study, resume)
    write_subconfigs(config, destination)
    atomic_write(destination / "study.json", study.model_dump_json(indent=2).encode())
    atomic_write(destination / "rl.json", config.model_dump_json(indent=2).encode())
    atomic_write(
        destination / "protocol.json",
        json.dumps(
            {
                "config_sha256": study.fingerprint(),
                "lag": study.lag,
                "warmup_updates": study.lag,
                "max_steps": study.max_steps,
                "exact_staleness_updates": study.max_steps - study.lag,
                "responses_per_update": study.response_batch_size,
                "budget_includes_bootstrap": True,
                "filter_zero_advantages": False,
                "all_zero_cohort": "optimizer_step_with_zero_policy_gradient",
                "generation_method": "version_indexed_delayed_cohorts",
                "warmup_age": 0,
                "steady_state_age": study.lag,
                "loss": "clipped GRPO using original behavior log-probabilities",
                "loss_reduction": study.loss_reduction,
                "reference_kl_coefficient": study.reference_kl_coefficient,
                "updates_per_cohort": study.updates_per_cohort,
                "advantage_normalization": study.advantage_normalization,
                "prompt_order": "one seeded shuffle, then cyclic traversal",
            },
            indent=2,
        ).encode(),
    )
    return config
