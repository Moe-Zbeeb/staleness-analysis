import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


GPU_PROBE_TIMEOUT_SECONDS = 300
GPU_PROBE_PARENT_MARGIN_SECONDS = 60
COLLECTIVE_HEALTH_TIMEOUT_SECONDS = 900
TORCH_IMPORT_TIMEOUT_SECONDS = 180
TORCH_IMPORT_TRACE_AFTER_SECONDS = 30

TORCH_IMPORT = """
import faulthandler
import json
import os
import sys
import time

started = time.monotonic()

def stage(name):
    print('[torch-import-stage]' + json.dumps({'stage': name, 'seconds': time.monotonic() - started}), file=sys.stderr, flush=True)

faulthandler.dump_traceback_later(int(os.environ['DEEPSEEK_STUDY_TORCH_IMPORT_TRACE_AFTER']), repeat=True)
stage('before_torch_import')
import torch
stage('after_torch_import')
faulthandler.cancel_dump_traceback_later()
print(json.dumps({'torch_version': torch.__version__, 'torch_file': torch.__file__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}), flush=True)
"""


def startup_deadlines():
    return {
        "torch_import_seconds": TORCH_IMPORT_TIMEOUT_SECONDS,
        "torch_import_trace_after_seconds": TORCH_IMPORT_TRACE_AFTER_SECONDS,
        "gpu_probe_seconds": GPU_PROBE_TIMEOUT_SECONDS,
        "gpu_probe_parent_margin_seconds": GPU_PROBE_PARENT_MARGIN_SECONDS,
        "gpu_probe_parent_seconds": GPU_PROBE_TIMEOUT_SECONDS + GPU_PROBE_PARENT_MARGIN_SECONDS,
        "collective_health_seconds": COLLECTIVE_HEALTH_TIMEOUT_SECONDS,
    }


def warm_torch_import(receipt, environment=None):
    receipt = Path(receipt)
    if receipt.exists():
        raise FileExistsError("Torch import receipt already exists; use a fresh launch attempt")
    environment = {
        **(os.environ if environment is None else environment),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "DEEPSEEK_STUDY_TORCH_IMPORT_TRACE_AFTER": str(TORCH_IMPORT_TRACE_AFTER_SECONDS),
    }
    record = {
        "stage": "warm_torch_import",
        "status": "failed",
        "interpreter": sys.executable,
        "job_id": environment.get("SLURM_JOB_ID"),
        "node": environment.get("SLURMD_NODENAME"),
        "timeout_seconds": TORCH_IMPORT_TIMEOUT_SECONDS,
        "trace_after_seconds": TORCH_IMPORT_TRACE_AFTER_SECONDS,
        "cuda_visible_devices": "",
        "omp_num_threads": "1",
    }
    stdout, stderr = "", ""
    print(
        json.dumps({"stage": record["stage"], "status": "starting", "timeout_seconds": TORCH_IMPORT_TIMEOUT_SECONDS}),
        flush=True,
    )
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", TORCH_IMPORT],
            env=environment,
            capture_output=True,
            text=True,
            timeout=TORCH_IMPORT_TIMEOUT_SECONDS,
        )
        stdout, stderr = result.stdout, result.stderr
        record["returncode"] = result.returncode
        if result.returncode:
            record["error"] = f"Torch import process exited with status {result.returncode}"
        else:
            try:
                details = json.loads(stdout.splitlines()[-1])
                if (
                    not isinstance(details, dict)
                    or details.get("cuda_visible_devices") != ""
                    or not isinstance(details.get("torch_version"), str)
                    or not details["torch_version"]
                    or not isinstance(details.get("torch_file"), str)
                    or not details["torch_file"]
                ):
                    raise ValueError("Incomplete Torch import result")
                record.update(status="complete", result=details)
            except (ValueError, IndexError) as error:
                record["error"] = f"Invalid Torch import acknowledgement: {error}"
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or "", error.stderr or ""
        record.update(status="timeout", error="Torch import exceeded its startup deadline")
    except OSError as error:
        record["error"] = f"Torch import process could not start: {type(error).__name__}: {error}"
    stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
    stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    stages = []
    for line in stderr.splitlines():
        if line.startswith("[torch-import-stage]"):
            try:
                stage = json.loads(line.removeprefix("[torch-import-stage]"))
            except ValueError:
                continue
            if isinstance(stage, dict) and isinstance(stage.get("stage"), str):
                stages.append(stage)
    record.update(
        elapsed_seconds=time.monotonic() - started,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-8000:],
        stages=stages[-32:],
        last_stage=stages[-1]["stage"] if stages else None,
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write(receipt, record)
    print(json.dumps({"stage": record["stage"], "status": record["status"], "receipt": str(receipt)}), flush=True)
    if record["status"] != "complete":
        raise RuntimeError(f"Torch import warmup failed; see {receipt}: {record['error']}")
    return record


def select_devices(probes, allocated_count, participating_count):
    allocated, healthy = probes["allocated_devices"], probes["healthy_devices"]
    if len(allocated) != allocated_count or len(set(allocated)) != allocated_count:
        raise ValueError("Unexpected Slurm allocation")
    if any("GPU is already occupied:" in row.get("error", "") for row in probes["results"]):
        raise ValueError("An allocated GPU is occupied by another process")
    if len(healthy) != participating_count or len(set(healthy)) != participating_count:
        raise ValueError("GPU health does not match the prepared topology")
    if not set(healthy).issubset(allocated):
        raise ValueError("Probe selected a GPU outside the allocation")
    if allocated_count != participating_count and (allocated_count, participating_count) != (8, 7):
        raise ValueError("Unassigned healthy GPUs are not allowed")
    records = {row["device"]: row for row in probes["results"]}
    if any(not records[device]["healthy"] for device in healthy):
        raise ValueError("Selected GPU failed its probe")
    uuids = [records[device]["uuid"] for device in healthy]
    if "unavailable" in uuids or len(uuids) != len(set(uuids)):
        raise ValueError("Participating GPU identities are unavailable or duplicated")
    return healthy


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def validate_node(manifest, environment):
    nodes = manifest.get("allowed_nodes", [manifest["node"]] if "node" in manifest else [])
    if not isinstance(nodes, list) or not nodes or any(not isinstance(node, str) for node in nodes):
        raise ValueError("Invalid prepared node allowlist")
    node = environment["SLURMD_NODENAME"]
    if node not in nodes or int(environment.get("SLURM_RESTART_COUNT", "0")):
        raise ValueError("Unexpected node or unsafe automatic restart")
    return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control.resolve()
    manifest = json.loads((control / "full-run.json").read_text())
    for name, expected in {"study.json": manifest["study_sha256"], **manifest["scripts_sha256"]}.items():
        if hashlib.sha256((control / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Full-run launch artifact changed: {name}")
    if manifest.get("startup_deadlines", startup_deadlines()) != startup_deadlines():
        raise ValueError("Full-run startup deadlines changed")
    node = validate_node(manifest, os.environ)
    release = Path(manifest["release"])
    sys.path.insert(0, str(release / "src"))
    from deepseek_study.config import StudyConfig
    from deepseek_study.runtime.identity import capture

    study = StudyConfig.read(control / "study.json")
    if study.fingerprint() != manifest["config_sha256"]:
        raise ValueError("Full-run configuration changed")
    if capture(release, study)["sha256"] != manifest["identity_sha256"]:
        raise ValueError("Frozen source, runtime or dataset changed")
    if study.output_dir.exists():
        raise FileExistsError("Refusing to overwrite or silently restart an existing run")
    job = control / f"job-{os.environ['SLURM_JOB_ID']}"
    job.mkdir(exist_ok=False)
    warm_torch_import(job / "import-warmup.json")
    allocated = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    subprocess.run(
        [
            sys.executable,
            str(control / "probe_allocated_gpus.py"),
            "--output",
            str(job / "device-probes.json"),
            "--timeout",
            str(GPU_PROBE_TIMEOUT_SECONDS),
        ],
        check=True,
        timeout=GPU_PROBE_TIMEOUT_SECONDS + GPU_PROBE_PARENT_MARGIN_SECONDS,
    )
    probes = json.loads((job / "device-probes.json").read_text())
    if probes["allocated_devices"] != allocated:
        raise ValueError("Probe differs from the current allocation")
    count = study.trainer_gpus + study.inference_gpus
    visible = select_devices(probes, manifest["allocated_gpus"], count)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    os.environ["PYTHONPATH"] = str(release / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={count}",
            str(release / "scripts/gpu_health.py"),
            "--expected-gpus",
            str(count),
            "--receipt",
            str(job / "hardware.json"),
        ],
        check=True,
        timeout=COLLECTIVE_HEALTH_TIMEOUT_SECONDS,
    )
    hardware = json.loads((job / "hardware.json").read_text())
    if hardware["world_size"] != count or len(hardware["devices"]) != count:
        raise ValueError("Collective world size differs from the run")
    if not all("A100" in row["name"] and row["bytes"] >= manifest["minimum_gpu_bytes"] for row in hardware["devices"]):
        raise ValueError("Allocated GPU hardware differs from the prepared run")
    command = [sys.executable, "-m", "deepseek_study.cli", "run", str(control / "study.json")]
    write(
        job / "launch.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": node,
            "allocated_devices": allocated,
            "visible_devices": visible,
            "unused_devices": [device for device in allocated if device not in visible],
            "inference_devices": visible[: study.inference_gpus],
            "trainer_devices": visible[study.inference_gpus :],
            "maximum_updates": study.max_steps,
            "lag": study.lag,
            "command": command,
            "config_sha256": manifest["config_sha256"],
            "identity_sha256": manifest["identity_sha256"],
            "startup_deadlines": startup_deadlines(),
        },
    )
    os.chdir(release)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
