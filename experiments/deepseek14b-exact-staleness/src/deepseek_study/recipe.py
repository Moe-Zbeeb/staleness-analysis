from pathlib import Path

from deepseek_study.config import StudyConfig


def baseline(lag, profile="80gb", max_steps=1000, seed=42, root=None):
    root = Path(root or "/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study")
    if profile not in {"80gb", "40gb"}:
        raise ValueError("Unknown hardware profile")
    small = profile == "40gb"
    return StudyConfig(
        model_path="/mnt/nfs/home/mohamadzbib/projects/models/exact-age-14b/deepseek-r1-distill-qwen-14b",
        dataset_path="/mnt/nfs/home/mohamadzbib/projects/exact-age-14b/release/deepscaler/data/train.parquet",
        data_manifest=root / "assets" / "train-manifest.json",
        prepared_model_path=root / "assets" / "native-model",
        output_dir=root / "outputs" / f"exact{lag}-{profile}-seed{seed}",
        lag=lag,
        prompts_per_update=64,
        responses_per_prompt=8,
        prompt_max_tokens=2048,
        response_max_tokens=8192,
        max_steps=max_steps,
        learning_rate=1e-6,
        lr_warmup_steps=30,
        lr_decay_steps=0,
        min_learning_rate=0.0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        clip_epsilon=0.2,
        advantage_normalization="centered",
        advantage_epsilon=1e-8,
        temperature=1.0,
        loss_reduction="global_token_mean",
        reference_kl_coefficient=0.0,
        updates_per_cohort=1,
        prompt_instruction="Please reason step by step, and put your final answer within \\boxed{}.",
        truncated_reward="grade_final",
        seed=seed,
        checkpoint_interval=25,
        trainer_gpus=7 if small else 4,
        inference_gpus=1 if small else 4,
        inference_tensor_parallel=1,
        rollout_concurrency=8 if small else 64,
        inference_max_sequences=2 if small else 16,
        trainer_micro_batch_size=1,
        trainer_attention="fa2",
        trainer_compile=False,
        optimizer_cpu_offload=False,
        activation_checkpointing=True,
        activation_cpu_offload=small,
        lm_head_chunk_size=1024 if small else 8192,
        inference_memory_utilization=0.9,
    )
