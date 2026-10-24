"""Command line entry point: download, train, all."""

from __future__ import annotations

import argparse

from . import config, data, train


def _print_summary(metrics: dict) -> None:
    ranking = metrics["test_ranking"]
    budget = metrics["operating_points"]["budget"]
    share = budget["fraction"]
    print(f"selected model : {metrics['selected_model']}")
    print(f"test PR-AUC    : {ranking['pr_auc']:.4f}")
    print(f"test ROC-AUC   : {ranking['roc_auc']:.4f}")
    print(f"test Brier     : {ranking['brier']:.4f}")
    print(f"precision@{share:.0%}    : {budget['test']['precision']:.4f}")
    print(f"recall@{share:.0%}       : {budget['test']['recall']:.4f}")
    print(f"f1@{share:.0%}           : {budget['test']['f1']:.4f}")
    print(f"report         : {config.REPORT_MD}")
    print(f"metrics        : {config.METRICS_JSON}")
    print(f"model          : {config.MODEL_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bank_marketing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and checksum the raw dataset")
    download.add_argument("--force", action="store_true")

    subparsers.add_parser("train", help="run CV, fit the final model, write the report")

    run_all = subparsers.add_parser("all", help="download + train")
    run_all.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "download":
        path = data.download_raw(force=args.force)
        print(f"raw dataset ready: {path}")
        return 0

    if args.command == "train":
        metrics = train.run(force_download=False)
        _print_summary(metrics)
        return 0

    metrics = train.run(force_download=args.force)
    _print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
