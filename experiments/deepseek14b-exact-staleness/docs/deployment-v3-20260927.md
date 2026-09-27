# v3 cluster deployment

The source fixes were pushed directly to GitHub main in `f369137be8e7f505ca588038bf59382e7403bb47`. The cluster's `current` symlink now selects `/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/releases/correctness-v3-20260927`. All 111 packaged files match the release inventory. The official PrimeRL checkout remains at the pinned commit with no tracked changes.

The older root working directory differs from both the previous frozen release and GitHub baseline. Those unexpected edits were preserved. Use the new release explicitly, including its source import path, rather than the legacy root's editable Python package:

```bash
export STUDY_RELEASE=/mnt/nfs/home/mohamadzbib/projects/deepseek14b-deepscaler-study/current
export PYTHONPATH="$STUDY_RELEASE/src"
"$STUDY_RELEASE/vendor/prime-rl/.venv/bin/python" -m deepseek_study.cli --help
```

This command does not launch training. Old run outputs, frozen releases and launch controls remain unchanged. New configurations must use the v3 manifest and new output directories; old small-model controls must be regenerated against this release.

Cluster validation confirmed package hashes, the prepared manifest body hash, matching grader dependency versions and successful clock/plain-answer grading probes. The combined asset/configuration preflight exceeded its 180-second deadline. A narrowed 55-second diagnostic showed Python waiting in `importlib.get_data` while loading NFS-hosted standard-library files during the `asyncio` import. This indicates an import/I/O delay in that diagnostic, not an observed GRPO failure; its wider cause and duration remain unmeasured.

The full cluster asset/runtime preflight and GPU pilot are therefore still unverified. No training was submitted, and the stopped studies remain stopped. Local validation and saved-response replays are detailed in the [hardening report](hardening-20260927.md).

The immutable package SHA256 is `6cf2cd092acf65c79c7eaab55d898e1f408518db1a6568d162bd3c80a8c5793c`. The post-deployment receipts are [deployment](../diagnostics/deployment-v3-20260927.json) and [cluster checks](../diagnostics/hardening-v3-cluster-preflight-20260927.json). These supplemental records were added after freezing the source release; they do not change its training source or manifest.
