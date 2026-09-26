import argparse
import asyncio
import json
from pathlib import Path

from deepseek_study.config import StudyConfig


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema")
    init = commands.add_parser("init")
    init.add_argument("destination", type=Path)
    init.add_argument("--lag", type=int, required=True)
    init.add_argument("--profile", choices=["80gb", "40gb"], default="80gb")
    init.add_argument("--max-steps", type=int, default=1000)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--root", type=Path)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("directory", type=Path)
    tracking = commands.add_parser("track")
    tracking.add_argument("directory", type=Path)
    tracking.add_argument("--once", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    for name in ("build", "run", "controller", "check", "prepare-data"):
        command = commands.add_parser(name)
        command.add_argument("study", type=Path)
        if name in {"build", "controller"}:
            command.add_argument("destination", type=Path)
        if name in {"run", "build", "controller"}:
            command.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        from deepseek_study.recipe import baseline

        study = baseline(args.lag, args.profile, args.max_steps, args.seed, args.root)
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        with args.destination.open("x") as stream:
            stream.write(study.model_dump_json(indent=2) + "\n")
        print(args.destination.resolve())
        return
    if args.command == "audit":
        from deepseek_study.rollouts.audit import audit

        print(json.dumps(audit(args.directory), indent=2))
        return
    if args.command == "schema":
        print(json.dumps(StudyConfig.model_json_schema(), indent=2))
        return
    if args.command == "track":
        from deepseek_study.tracking.runboard import observe

        print(json.dumps(observe(args.directory, once=args.once), indent=2))
        return
    if args.command == "prepare":
        from deepseek_study.dataset.assets import prepare

        print(json.dumps(prepare(args.model, args.dataset, args.destination, args.manifest), indent=2))
        return
    study = StudyConfig.read(args.study)
    if args.command == "prepare-data":
        from deepseek_study.dataset.prepare import prepare_data

        print(json.dumps(asyncio.run(prepare_data(study)), indent=2))
    elif args.command == "build":
        from deepseek_study.runtime.build import build

        build(study, args.destination, args.resume)
        print(args.destination.resolve())
    elif args.command == "check":
        from deepseek_study.dataset.assets import validate_prepared
        from deepseek_study.runtime.build import resolve

        resolve(study)
        print(json.dumps(validate_prepared(study), indent=2))
    elif args.command == "run":
        from deepseek_study.runtime.launcher import launch

        print(launch(study, Path(__file__).resolve().parents[2], args.resume))
    elif args.command == "controller":
        from deepseek_study.rollouts.controller import control

        asyncio.run(control(study, args.destination, args.resume))


if __name__ == "__main__":
    main()
