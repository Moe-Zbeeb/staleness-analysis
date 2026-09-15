# Objective and staleness semantics

For each prompt, the orchestrator samples a group of `G = 8` completions. If completion `i` receives verifier reward `R_i`, PrimeRL's GRPO orchestrator assigns the same scalar advantage to every generated action token in that completion:

```text
A_i = R_i - mean_j(R_j)
```

The implementation does not divide by the group reward standard deviation and does not apply a length penalty.

For token `t` from completion `i`, define:

```text
log_ratio_i,t = log pi_theta(y_i,t | x, y_i,<t) - log mu_i,t
ratio_i,t = exp(log_ratio_i,t)
clipped_i,t = clip(ratio_i,t, 0.8, 1.2)
```

`mu` is the inference policy that generated the rollout. `max_off_policy_steps = 2` permits a rollout to trail the current trainer policy by at most two accepted policy versions.

The optimized loss is:

```text
L(theta) = -(1 / N) sum_(i,t in action tokens)
           min(ratio_i,t A_i, clipped_i,t A_i)
```

`N` is the global number of eligible RL action tokens across data- and context-parallel ranks. This is global token averaging, not equal per-completion averaging, so longer completions contribute more aggregate token weight.

There is no reference-policy KL penalty, entropy bonus, critic, or value loss. `approx_kl = ratio - log_ratio - 1` is telemetry only.

This custom loss bypasses PrimeRL's default DPPO plus squared-log-ratio KL loss. The default loss in the pinned runtime would use direction-dependent probability-difference masking and add `1e-3 * log_ratio^2`; it is not active in these configs.

Staleness is controlled independently of the loss:

- `max_off_policy_steps = 0` accepts only rollouts from the current policy version.
- `max_off_policy_steps = 1` accepts the current and immediately previous permitted version under PrimeRL's version accounting.
- `max_off_policy_steps = 2` is the validated experiment setting.
- Larger values increase throughput tolerance but broaden the behavior/trainer mismatch measured by the importance ratio and approximate KL.

