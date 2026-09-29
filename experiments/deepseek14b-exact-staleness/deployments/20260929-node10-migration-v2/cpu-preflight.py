import json
import shutil
from pathlib import Path
from deepseek_study.config import StudyConfig
from deepseek_study.runtime.identity import capture

control = Path(__file__).parent
spec = json.loads((control / "storage-spec.json").read_text())
workspace = Path(spec["workspace"])
ready = json.loads((workspace / "ready.json").read_text())
study = StudyConfig.read(workspace / "study.json")
identity = capture(workspace / "release", study)
assert identity["sha256"] == json.loads((control / "source-identity.json").read_text())["sha256"]
assert study.fingerprint() == StudyConfig.read(control / "target-study.json").fingerprint()
assert study.checkpoint_keep_last == 3 and study.checkpoint_keep_interval == 1000
assert spec["require_verified_shared_milestones"] is True
free = shutil.disk_usage(workspace).free
minimum = 180 * 1024**3
if free < minimum:
    raise RuntimeError(f"Insufficient post-staging local capacity: {free} < {minimum}")
result = {"runtime_and_source_identity_match": True, "identity_sha256": identity["sha256"], "study_sha256": ready["study_sha256"], "node": ready["node"], "gpu_execution_verified": False, "local_free_bytes": free, "required_free_before_resume_bytes": minimum, "local_keep_last": 3, "local_keep_interval": 1000, "shared_milestones": 100, "require_verified_shared_milestones": True}
(control / "cpu-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
