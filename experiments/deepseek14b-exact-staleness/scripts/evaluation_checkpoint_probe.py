"""Read-only compatibility check for a training checkpoint's tensor schema."""
import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoModelForCausalLM

from deepseek_study.evaluation.common import file_hash, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trainer", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", dtype=torch.bfloat16)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    metadata = FileSystemReader(args.trainer).read_metadata()
    actual = {k.removeprefix("app.model."): v for k, v in metadata.state_dict_metadata.items()
              if k.startswith("app.model.")}
    if set(actual) != set(expected) or any(tuple(actual[k].size) != expected[k] for k in expected):
        raise ValueError("Training checkpoint does not match the HF model schema")
    receipt = {"schema_matches": True, "model_tensors": len(actual),
               "parameters": sum(v.numel() for v in model.state_dict().values()),
               "stored_dtypes": sorted({str(v.properties.dtype) for v in actual.values()}),
               "metadata_sha256": file_hash(args.trainer / ".metadata"),
               "scope": "Tensor names/shapes only; does not certify checkpoint completeness or numerical parity"}
    write(args.receipt, receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
