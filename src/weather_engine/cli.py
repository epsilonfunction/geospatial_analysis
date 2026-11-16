"""Command-line workflows. Network access only occurs in explicitly requested ingestion."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from .data import Archive, atomic_write, code_version, content_hash, json_bytes, normalize_records, utc


def clean_json(value):
    if is_dataclass(value):
        return clean_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return clean_json(value.to_dict("records"))
    if isinstance(value, (pd.Timestamp, date, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def emit(value):
    print(json_bytes(clean_json(value)).decode())


def load_config(path=None):
    default = Path(__file__).resolve().parents[2] / "configs" / "singapore.json"
    if path is None and not default.exists():
        return {"mode": "strict", "required_variables": ["temperature_c"], "max_forecast_age_hours": 24}
    config = json.loads(Path(path or default).read_text("utf-8"))
    if config.get("mode", "strict") not in {"strict", "reconstructed"}:
        raise ValueError("Unknown replay mode")
    return config


def _comma(text, kind=str):
    return [kind(item.strip()) for item in text.split(",") if item.strip()]


def _ingest(args, archive, config):
    from .sources import fetch_ecmwf, fetch_nea, fetch_noaa
    variables = _comma(args.variables) if args.variables else ["temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms", "wind_direction_deg"]
    completed = set()
    while True:
        now = utc()
        failures = []
        if args.source == "nea":
            for variable in variables:
                try:
                    emit(archive.ingest(fetch_nea(variable)))
                except Exception as exc:
                    failures.append({"variable": variable, "error": f"{type(exc).__name__}: {exc}"})
                time.sleep(0.4)  # Remain below the unauthenticated API burst limit.
        else:
            leads = _comma(args.leads, int) if args.leads else config.get("forecast_leads", list(range(0, 73, 3)))
            members = _comma(args.members) if args.members else (["0"] if args.source == "gfs" else config.get(f"{args.source}_members", ["0"]))
            # A watch catches up missing cycles within the freshness window, oldest first.
            newest = now.floor("6h")
            cycles = [utc(args.cycle)] if args.cycle else ([newest - pd.Timedelta(hours=i) for i in [18, 12, 6, 0]] if args.watch else [newest - pd.Timedelta(hours=6)])
            for cycle in cycles:
                task = content_hash(json_bytes({"source": args.source, "cycle": cycle, "leads": leads, "members": members}))
                receipt = archive.root / "collection" / f"{task}.json"
                if args.watch and (task in completed or receipt.exists()):
                    continue
                try:
                    fetch = fetch_ecmwf if args.source == "ecmwf" else fetch_noaa
                    batches = fetch(cycle=cycle, leads=leads, members=members) if args.source == "ecmwf" else fetch(model=args.source, cycle=cycle, leads=leads, members=members)
                    receipts = [archive.ingest(batch) for batch in batches]
                    emit({"source": args.source, "cycle": cycle, "batches": receipts})
                    # A successful fetch is complete only if its adapter explicitly says so.
                    if batches and all(batch.metadata.get("complete", True) for batch in batches):
                        atomic_write(receipt, json_bytes({"source": args.source, "cycle": cycle, "completed_at": utc(), "batch_ids": [r["batch_id"] for r in receipts]}))
                        completed.add(task)
                except Exception as exc:
                    failures.append({"source": args.source, "cycle": str(cycle), "error": f"{type(exc).__name__}: {exc}"})
        if failures:
            event = {"recorded_at": str(utc()), "source": args.source, "failures": failures}
            atomic_write(archive.root / "collection" / "failures" / f"{uuid4().hex}.json", json_bytes(event))
            emit(event)
        if not args.watch:
            return 1 if failures else 0
        interval = config.get("poll_seconds", 300) if args.source == "nea" else config.get("forecast_poll_seconds", 1800)
        time.sleep(max(1, interval))


def _backfill(args, archive):
    from .sources import parse_demand_csv, parse_grib, parse_nea_csv
    result = []
    for filename in args.files:
        if args.format == "nea-csv":
            if not args.variable:
                raise ValueError("NEA CSV imports require --variable")
            batch = parse_nea_csv(filename, args.variable, available_at=args.available_at)
        elif args.format == "demand-csv":
            batch = parse_demand_csv(filename, available_at=args.available_at, semantics=args.semantics)
        elif args.format == "grib":
            batch = parse_grib(filename, source=args.source, model=args.model, init_time=args.cycle,
                               available_at=args.available_at, availability_confidence="published" if args.available_at else "unknown")
        else:
            from .sources import SourceBatch
            path = Path(filename)
            frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            batch = SourceBatch(path.read_bytes(), path.name, {"source": args.source, "import_format": "canonical", "demo": args.demo}, normalize_records(frame))
        result.append(archive.ingest(batch))
    emit(result)
    return 0


def _train(args, archive, config):
    from .modeling import train_bundle
    records = archive.read()
    if records.empty:
        raise ValueError("Archive is empty; ingest or backfill records first")
    as_of = utc(args.as_of)
    bundle = train_bundle(records, as_of, config=config)
    name = args.name or as_of.strftime("model-%Y%m%dT%H%M%S")
    if Path(name).name != name:
        raise ValueError("Model name must be a single path component")
    target = archive.root / "models" / f"{name}.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        joblib.dump(bundle, stream)
    from .data import point_in_time
    known = point_in_time(records, as_of, config.get("mode", "strict"), config.get("latency_hours"))
    atomic_write(target.with_suffix(".json"), json_bytes({"as_of": str(as_of), "config": config,
                 "model_checksum": content_hash(target.read_bytes()), "snapshot": archive.snapshot(known), "code_version": code_version()}))
    emit({"model": str(target), "as_of": str(as_of)})
    return 0


def _backtest(args, archive, config):
    from .evaluation import audit_coverage, run_backtest
    records = archive.read()
    if records.empty:
        raise ValueError("Archive is empty; import historical forecasts and observations first")
    if args.mode:
        config["mode"] = args.mode
    if args.latency_hours is not None:
        config["latency_hours"] = args.latency_hours
    coverage = audit_coverage(records, start=args.start, end=args.end)
    if args.audit_only:
        emit(coverage)
        return 0
    result = run_backtest(records, args.start, args.end, config=config)
    run_id = args.run_id or "backtest-" + utc().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
    predictions = result["predictions"]
    scores = result.get("scores", pd.DataFrame())
    manifest = clean_json({"kind": "backtest", "mode": config.get("mode", "strict"), "config": config,
                          "start": args.start, "end": args.end, "coverage": coverage,
                          "snapshot": archive.snapshot(records), **result.get("manifest", {})})
    path = archive.save_run(run_id, predictions, manifest, scores=scores, snapshot=records)
    for name, value in result.items():
        if name in {"predictions", "scores", "manifest"}:
            continue
        if isinstance(value, pd.DataFrame):
            value.to_parquet(path / f"{name}.parquet", index=False)
        else:
            atomic_write(path / f"{name}.json", json_bytes(clean_json(value)))
    emit({"run": str(path), "prediction_rows": len(predictions), "score_rows": len(scores)})
    return 0


def _forward(args, archive, config):
    from .forward import replay_forward, run_forward, score_forward
    if args.replay:
        replayed = replay_forward(archive, args.replay)
        original = pd.read_parquet(archive.root / "runs" / args.replay / "predictions.parquet")
        pd.testing.assert_frame_equal(replayed.reset_index(drop=True), original.reset_index(drop=True), check_dtype=False)
        emit({"run_id": args.replay, "replay_equal": True, "rows": len(replayed)})
        return 0
    if args.score:
        emit({"verification": str(score_forward(archive, args.score))})
        return 0
    if config.get("mode", "strict") != "strict":
        raise ValueError("Forward testing always requires strict observed availability")
    first = True
    while True:
        emit(run_forward(archive, config, model_path=args.model, new_session=args.new_session and first))
        first = False
        if not args.watch:
            return 0
        time.sleep(30)


def _report(args, archive):
    from .evaluation import audit_coverage
    if args.run_id:
        directory = archive.root / "runs" / args.run_id
        result = json.loads((directory / "manifest.json").read_text("utf-8"))
        score_path = directory / "scores.parquet"
        if score_path.exists():
            result["scores"] = pd.read_parquet(score_path)
        emit(result)
    else:
        runs = []
        for path in sorted((archive.root / "runs").glob("*/manifest.json")):
            if path.parent.name.startswith("."):
                continue
            manifest = json.loads(path.read_text("utf-8"))
            runs.append({k: manifest.get(k) for k in ["run_id", "kind", "mode", "phase", "timely", "error"]})
        emit({"coverage": audit_coverage(archive.read()), "runs": runs})
    return 0


def parser():
    cli = argparse.ArgumentParser(description="Singapore weather research with immutable data vintages")
    cli.add_argument("--data-dir", default="data/weather_engine")
    cli.add_argument("--config", help="JSON configuration (defaults to configs/singapore.json)")
    commands = cli.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Collect live NEA or a NOAA/ECMWF cycle")
    ingest.add_argument("--source", required=True, choices=["nea", "gfs", "gefs", "ecmwf"])
    ingest.add_argument("--cycle", help="Timezone-aware initialization time; default previous 6-hour cycle")
    ingest.add_argument("--variables", help="Comma-separated NEA canonical variables")
    ingest.add_argument("--leads", help="Comma-separated forecast hours, max 72")
    ingest.add_argument("--members", help="Comma-separated ensemble member numbers; omitted uses configured full ensemble")
    ingest.add_argument("--watch", action="store_true", help="Poll this source; run separately from the forward scheduler")
    backfill = commands.add_parser("backfill", help="Import original archive GRIB, NEA CSV, or actual-demand CSV files")
    backfill.add_argument("files", nargs="+")
    backfill.add_argument("--format", required=True, choices=["grib", "nea-csv", "demand-csv", "canonical"])
    backfill.add_argument("--source", default="noaa")
    backfill.add_argument("--model", default="gfs")
    backfill.add_argument("--variable")
    backfill.add_argument("--cycle")
    backfill.add_argument("--available-at", help="Verified timezone-aware publication timestamp; omit if unknown")
    backfill.add_argument("--semantics", default="actual_system_demand")
    backfill.add_argument("--demo", action="store_true", help="Mark canonical fixtures as synthetic demo data")
    train = commands.add_parser("train", help="Fit a versioned bundle from information available at a cutoff")
    train.add_argument("--as-of", required=True)
    train.add_argument("--name")
    backtest = commands.add_parser("backtest", help="Audit or replay daily decisions with monthly refits")
    backtest.add_argument("--start", default="2023-01-01")
    backtest.add_argument("--end", default="2024-12-31")
    backtest.add_argument("--mode", choices=["strict", "reconstructed"])
    backtest.add_argument("--latency-hours", type=float)
    backtest.add_argument("--audit-only", action="store_true")
    backtest.add_argument("--run-id")
    forward = commands.add_parser("forward", help="Freeze a live champion and record noon decisions")
    forward.add_argument("--model", help="Locally trained .joblib bundle; omitted uses baselines")
    forward.add_argument("--watch", action="store_true")
    forward.add_argument("--new-session", action="store_true")
    forward.add_argument("--replay", metavar="RUN_ID")
    forward.add_argument("--score", metavar="RUN_ID")
    report = commands.add_parser("report", help="Print coverage, experiments, or one experiment's results")
    report.add_argument("--run-id")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        archive = Archive(args.data_dir)
        config = load_config(args.config)
        if args.command == "ingest":
            return _ingest(args, archive, config)
        if args.command == "backfill":
            return _backfill(args, archive)
        if args.command == "train":
            return _train(args, archive, config)
        if args.command == "backtest":
            return _backtest(args, archive, config)
        if args.command == "forward":
            return _forward(args, archive, config)
        return _report(args, archive)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
