"""Command-line interface for NeSPReSO."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from nespreso.config import load_config
from nespreso.io.download.aviso import download_aviso
from nespreso.io.download.copernicus import download_ostia_sst, download_sss_range
from nespreso.runner import run_training


def parse_months(spec: str) -> list[int]:
    """Parse --months: inclusive range ``1-3`` or comma list ``1,2,12``."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start_s, end_s = spec.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start > end:
            raise ValueError(f"invalid month range: {spec}")
        months = list(range(start, end + 1))
    else:
        months = [int(part.strip()) for part in spec.split(",") if part.strip()]

    if not months:
        raise ValueError("empty months specification")
    for month in months:
        if not 1 <= month <= 12:
            raise ValueError(f"month out of range (1-12): {month}")
    return months


def _months_arg(value: str) -> list[int]:
    try:
        return parse_months(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_common_bbox(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)


def _add_date_range(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")


def _add_download_subparser(subparsers: argparse._SubParsersAction) -> None:
    dl = subparsers.add_parser("download", help="Download satellite products with date/bbox filters")
    dl_sub = dl.add_subparsers(dest="product", required=True)

    aviso = dl_sub.add_parser("aviso", help="AVISO SSH (year/month loops)")
    aviso.add_argument("--output", required=True)
    aviso.add_argument("--start-year", type=int, required=True)
    aviso.add_argument("--end-year", type=int, required=True)
    aviso.add_argument(
        "--months",
        type=_months_arg,
        default=None,
        help="Months to download: range '1-3' or list '1,2,12' (default: all)",
    )
    _add_common_bbox(aviso)

    ostia = dl_sub.add_parser("ostia", help="OSTIA SST via copernicusmarine (OISST replacement)")
    ostia.add_argument("--output", required=True)
    _add_date_range(ostia)
    _add_common_bbox(ostia)

    sss = dl_sub.add_parser("sss", help="SSS via copernicusmarine day loop")
    sss.add_argument("--output", required=True)
    _add_date_range(sss)
    _add_common_bbox(sss)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nespreso", description="NeSPReSO ocean ML pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/default.yaml in repo root)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Build dataset pickle, split, train, save checkpoint")
    train.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (overrides config monitor.tensorboard)",
    )
    train.add_argument("--log-dir", default=None, help="TensorBoard log directory")

    _add_download_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "train":
        if args.tensorboard:
            from dataclasses import replace

            log_dir = args.log_dir or cfg.monitor.log_dir
            cfg = replace(cfg, monitor=replace(cfg.monitor, tensorboard=True, log_dir=log_dir))
        run_training(cfg)
        return 0

    if args.command == "download":
        if args.product == "aviso":
            download_aviso(
                args.output,
                args.start_year,
                args.end_year,
                args.min_lon,
                args.max_lon,
                args.min_lat,
                args.max_lat,
                months=args.months,
            )
        elif args.product == "ostia":
            download_ostia_sst(
                args.output,
                datetime.fromisoformat(args.start),
                datetime.fromisoformat(args.end),
                args.min_lon,
                args.max_lon,
                args.min_lat,
                args.max_lat,
            )
        elif args.product == "sss":
            download_sss_range(
                args.output,
                datetime.fromisoformat(args.start),
                datetime.fromisoformat(args.end),
                args.min_lon,
                args.max_lon,
                args.min_lat,
                args.max_lat,
            )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
