import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def main():
    from deepseek_study.assets import prepare, validate_prepared
    from deepseek_study.build import build
    from deepseek_study.identity import capture
    from deepseek_study.recipe import baseline

    root = Path(__file__).resolve().parents[1]
    study = baseline(1, root=root)
    if not study.prepared_model_path.exists():
        print("Verifying pinned model and dataset files", flush=True)
        prepare(study.model_path, study.dataset_path, study.prepared_model_path, root / "manifests/model.json")
    validation = validate_prepared(study)
    fixtures = {
        root / "assets/tokenizer-fixture": study.model_path,
        root / "assets/dataset-fixture/data/train.parquet": study.dataset_path,
    }
    for path, target in fixtures.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if path.resolve() != target.resolve():
                raise ValueError(f"Unexpected fixture target: {path}")
        elif path.exists():
            raise ValueError(f"Refusing to replace fixture: {path}")
        else:
            path.symlink_to(target)
    result = root / "diagnostics/cluster"
    result.mkdir(parents=True, exist_ok=True)
    for profile in ("80gb", "40gb"):
        build(baseline(1, profile=profile, root=root), result / f"resolved-{profile}")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests", "--disable-warnings", "--tb=short"],
        cwd=root,
        check=True,
    )
    receipt = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "assets": validation,
        "identity": capture(root, study),
        "gpu_execution_verified": False,
        "study_training_started": False,
    }
    (result / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
