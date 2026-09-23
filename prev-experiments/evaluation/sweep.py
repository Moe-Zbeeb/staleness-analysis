import argparse
from pathlib import Path

from math_sweep.core import CONFIG, ROOT, digest, file_hash, load_catalogs, make_plan, read_json, read_jsonl, validate_records, write_json


def main():
    parser = argparse.ArgumentParser(description="Separate, manifest-bound math evaluation; never submits training or Slurm jobs.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config-dir", type=Path, default=CONFIG)
    plan.add_argument("--models", nargs="+")
    plan.add_argument("--profiles", nargs="+")
    plan.add_argument("--output", type=Path, required=True)
    data = commands.add_parser("prepare-data")
    data.add_argument("--config-dir", type=Path, default=CONFIG)
    data.add_argument("--data-root", type=Path, default=ROOT)
    data.add_argument("--retained-path", type=Path)
    data.add_argument("--training-path", type=Path)
    data.add_argument("--decisions", type=Path)
    data.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--cell", required=True)
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--tokenizer-root", type=Path, default=ROOT)
    run = commands.add_parser("run")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--prepared-root", type=Path, required=True)
    report.add_argument("--output-root", type=Path, required=True)
    report.add_argument("--report", type=Path, required=True)
    report.add_argument("--allow-partial", action="store_true")
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    commands.add_parser("validate-grader")
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(*load_catalogs(args.config_dir), selected_models=args.models, selected_profiles=args.profiles)
        write_json(args.output, result)
        print(result["summary"])
    elif args.command == "prepare-data":
        from math_sweep.data import prepare_data
        catalog = read_json(args.config_dir / "benchmarks.json")
        result = prepare_data(catalog, root=args.data_root, retained_path=args.retained_path, training_path=args.training_path,
                              decisions=read_json(args.decisions) if args.decisions else None)
        write_json(args.output, result)
        write_json(args.output.with_suffix(".receipt.json"), {"sha256": file_hash(args.output), "catalog_sha256": digest(catalog)})
        print(f"Prepared {len(result['rows'])} rows; inspect overlap_audit before preparing GPU cells.")
    elif args.command == "prepare":
        from math_sweep.prepare import prepare_cell
        cells = [cell for cell in read_json(args.plan)["cells"] if cell["cell_id"] == args.cell]
        if len(cells) != 1:
            raise ValueError("Select one exact cell ID from the frozen plan")
        print(prepare_cell(cells[0], args.data, args.output_dir, args.tokenizer_root))
    elif args.command == "run":
        from math_sweep.runner import run_cell
        print(run_cell(args.prepared, args.output_root))
    elif args.command == "validate-grader":
        from math_sweep.scoring import validate_grader
        print(validate_grader())
    else:
        report_results(args)


def report_results(args):
    from math_sweep.scoring import render_markdown, summarize
    records = []
    missing = []
    provenance = []
    runtime = {"elapsed_seconds": 0.0, "gpu_hours": 0.0, "timing_complete": True}
    cell_files = sorted(args.prepared_root.glob("*/cell.json"))
    if not cell_files:
        raise ValueError("No prepared cells found")
    for cell_file in cell_files:
        cell = read_json(cell_file)
        directory = args.output_root / cell["cell_id"]
        receipt_path = directory / "receipt.json"
        if not receipt_path.exists() or read_json(receipt_path).get("status") != "complete":
            missing.append({"cell_id": cell["cell_id"], "model": cell["model_id"], "profile": cell["profile"], "benchmark": cell["benchmark"], "budget": cell["budget"]})
            continue
        receipt = read_json(receipt_path)
        path = directory / "records.jsonl"
        for key in ("cell_id", "comparison_sha256", "expected_responses"):
            if receipt.get(key) != cell[key]:
                raise ValueError(f"Completion receipt identity mismatch: {key}")
        if receipt["status"] != "complete" or receipt["records_sha256"] != file_hash(path):
            raise ValueError(f"Invalid completion receipt: {directory}")
        if receipt["preparation_sha256"] != file_hash(cell_file.parent / "preparation.json"):
            raise ValueError("Completion belongs to a different preparation")
        if receipt["probes_sha256"] != file_hash(directory / "probes.json"):
            raise ValueError("Decoding probe receipt mismatch")
        probes = read_json(directory / "probes.json")
        if probes.get("status") != "passed" or probes.get("cell_id") != cell["cell_id"] or probes.get("preparation_sha256") != receipt["preparation_sha256"]:
            raise ValueError("Decoding probe identity or status mismatch")
        preparation = read_json(cell_file.parent / "preparation.json")
        if any(file_hash(cell_file.parent / name) != sha for name, sha in preparation["files"].items()):
            raise ValueError("Prepared inputs changed since completion")
        rows = read_jsonl(cell_file.parent / "prompts.jsonl")
        selected = read_jsonl(path)
        validate_records(selected, rows, cell, complete=True)
        if receipt.get("responses") != len(selected):
            raise ValueError("Receipt response count mismatch")
        records.extend(selected)
        provenance.append({"cell_id": cell["cell_id"], "receipt": str(receipt_path), "receipt_sha256": file_hash(receipt_path)})
        runtime["elapsed_seconds"] += receipt["elapsed_seconds"]
        runtime["gpu_hours"] += receipt["gpu_hours"]
        runtime["timing_complete"] &= receipt["timing_complete"]
    if missing and not args.allow_partial:
        raise ValueError(f"{len(missing)} prepared cells are incomplete; --allow-partial explicitly labels a partial report")
    if not records:
        raise ValueError("No completed, verified responses available")
    full = summarize(records, args.bootstrap_samples)
    heldout_records = [record for record in records if record["heldout"]]
    heldout = summarize(heldout_records, args.bootstrap_samples) if heldout_records else None
    result = {"status": "partial" if missing else "complete_for_prepared_cells", "missing_cells": missing,
              "provenance": provenance, "runtime": runtime, "full": full, "training_overlap_filtered": heldout}
    write_json(args.report, result)
    markdown = "# Math benchmark sweep\n\n" + f"Status: {result['status']}. Scope is the prepared cells, not all planned or pending models.\n\n"
    markdown += "## Full benchmark membership\n\n" + render_markdown(full)
    if heldout:
        markdown += "\n\n## Confirmed training overlaps excluded\n\n" + render_markdown(heldout)
    markdown += "\n\nFiltering covers the audited GRPO training data; it does not establish absence of pretraining overlap.\n"
    markdown += f"\nMeasured GPU-hours: {runtime['gpu_hours']:.3f}. Timing complete: {runtime['timing_complete']}. Elapsed seconds are summed across cells, not wall time for parallel execution.\n"
    args.report.with_suffix(".md").write_text(markdown)
    print(args.report)


if __name__ == "__main__":
    main()
