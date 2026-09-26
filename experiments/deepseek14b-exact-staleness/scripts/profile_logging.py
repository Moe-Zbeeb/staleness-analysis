import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = sorted(args.tokens.glob("rank_*.npz"))
    if not sources:
        raise ValueError("No archived token files found")
    for source in sources:
        started = time.perf_counter()
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        read_seconds = time.perf_counter() - started
        for repeat in range(args.repeats):
            buffer = io.BytesIO()
            started = time.perf_counter()
            np.savez_compressed(buffer, **arrays)
            compressed = time.perf_counter() - started
            target = args.output / f"{source.stem}-{repeat}.npz"
            started = time.perf_counter()
            with target.open("xb") as stream:
                stream.write(buffer.getbuffer())
                stream.flush()
                os.fsync(stream.fileno())
            write_seconds = time.perf_counter() - started
            with np.load(target, allow_pickle=False) as result:
                if set(result.files) != set(arrays):
                    raise AssertionError("Archive column set changed")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(result[key], value)
            rows.append(
                {
                    "source": str(source),
                    "repeat": repeat,
                    "read_and_decompress_seconds": read_seconds,
                    "compress_seconds": compressed,
                    "write_and_fsync_seconds": write_seconds,
                    "bytes": len(buffer.getbuffer()),
                    "tokens": len(arrays["token_id"]),
                    "exact_round_trip": True,
                }
            )
    receipt = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"), "rows": rows}
    (args.output / "timings.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
