import argparse
import json
import signal
import sys
from pathlib import Path

from .common import digest, immutable_json, read
from .discovery import discover, enqueue_evaluation, register
from .export import publish_baseline
from .protocol import Protocol, from_sweep
from .queue import Queue
from .runner import WorkerProfile
from .service import execute, initialize, serve, work


def main():
    parser = argparse.ArgumentParser(description="Independent, recoverable evaluation of completed study checkpoints")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("root", type=Path)
    init.add_argument("--tp", type=int, choices=[1, 2], default=1)
    init.add_argument("--max-sequences", type=int, default=2)
    freeze = commands.add_parser("freeze-sweep")
    for key in ("manifest", "policy", "model", "destination"):
        freeze.add_argument(key, type=Path)
    freeze.add_argument("--benchmarks", nargs="+", default=["math500", "aime24", "aime25"])
    reg = commands.add_parser("register")
    for key in ("root", "run", "protocol"):
        reg.add_argument(key, type=Path)
    for name in ("discover", "status"):
        commands.add_parser(name).add_argument("root", type=Path)
    collection = commands.add_parser("collect")
    collection.add_argument("roots", nargs="+", type=Path)
    coordinator = commands.add_parser("serve")
    coordinator.add_argument("root", type=Path)
    coordinator.add_argument("--once", action="store_true")
    coordinator.add_argument("--slurm-config", type=Path)
    worker = commands.add_parser("work")
    worker.add_argument("root", type=Path)
    worker.add_argument("--kind", choices=["export", "evaluate"], required=True)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--stall-seconds", type=int, default=1800)
    child = commands.add_parser("execute")
    child.add_argument("root", type=Path)
    child.add_argument("claim", type=Path)
    retry = commands.add_parser("retry")
    retry.add_argument("root", type=Path)
    retry.add_argument("task_id")
    baseline = commands.add_parser("baseline")
    for key in ("root", "model", "protocol"):
        baseline.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.command == "init":
        result = str(initialize(args.root, WorkerProfile(tensor_parallel=args.tp, max_sequences=args.max_sequences)))
    elif args.command == "freeze-sweep":
        result = from_sweep(args.manifest, args.policy, args.model, args.destination, args.benchmarks).identity
    elif args.command == "register":
        result = register(args.root, args.run, args.protocol)
    elif args.command == "discover":
        result = discover(args.root)
    elif args.command == "status":
        result = Queue(args.root).snapshot()
    elif args.command == "collect":
        from .report import collect

        result = collect(args.roots)
    elif args.command == "serve":
        result = serve(args.root, once=args.once, slurm=read(args.slurm_config) if args.slurm_config else None)
    elif args.command == "work":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        result = work(args.root, args.kind, once=args.once, stall_seconds=args.stall_seconds)
    elif args.command == "execute":
        result = execute(args.root, args.claim)
    elif args.command == "retry":
        result = Queue(args.root).retry(args.task_id)
    elif args.command == "baseline":
        protocol = Protocol.load(args.protocol)
        immutable_json(args.root / "protocols" / f"{protocol.identity}.json", protocol.model_dump())
        destination = args.root / "baselines" / digest(str(args.model.resolve()))
        receipt = publish_baseline(args.model, destination)
        protocol.validate_runtime(destination, backend=False)
        result = enqueue_evaluation(Queue(args.root), {
            "kind": "evaluate", "run_id": "baseline-" + receipt["checkpoint_id"], "step": 0, "lag": 0,
            "training_seed": None, "model": str(destination.resolve()), "checkpoint_id": receipt["checkpoint_id"],
            "protocol": protocol.identity,
        }, protocol)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
