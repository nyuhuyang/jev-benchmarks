from __future__ import annotations

import argparse

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "report", "anchor"):
        subcommand = commands.add_parser(command)
        subcommand.add_argument("--config", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--config", required=True)
    probe.add_argument("--static-only", action="store_true")
    fewshot = commands.add_parser("fewshot")
    fewshot.add_argument("--config", required=True)
    fewshot.add_argument("--backend", required=True, choices=("qwen_probe", "tfidf_lr", "prior"))
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument(
        "--backend",
        required=True,
        choices=(
            "gliner",
            "jev",
            "jev_openrouter",
            "laya_base",
            "laya_multilingual",
            "laya_typed",
            "qwen_logit",
        ),
    )
    run.add_argument(
        "--split",
        choices=("pilot", "calibration", "test", "permutation", "latency"),
        default="test",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "probe":
        from pathlib import Path

        from .probe import run_probe

        print(run_probe(Path(args.config), static_only=args.static_only))
        return
    config = load_config(args.config)
    if args.command == "prepare":
        from .data import prepare_manifest

        print(prepare_manifest(config))
    elif args.command == "fewshot":
        from .fewshot import run_fewshot

        print(run_fewshot(config, args.backend))
    elif args.command == "anchor":
        from .anchor import run_anchor

        print(run_anchor(config))
    elif args.command == "run":
        from .runner import run_backend

        if config.raw.get("schema_version") == 2:
            from .v2_runner import run_v2_backend

            print(run_v2_backend(config, args.backend, split=args.split))
        else:
            print(run_backend(config, args.backend))
    elif args.command == "report":
        from .report import build_report

        print("\n".join(str(path) for path in build_report(config)))
