"""Deadline-aware, immutable forward experiments; the collector runs separately."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from .data import Archive, atomic_write, code_version, content_hash, exclusive_lock, json_bytes, point_in_time, utc

SG = "Asia/Singapore"


def daily_deadline(day) -> pd.Timestamp:
    return pd.Timestamp(str(day)[:10], tz=SG).replace(hour=12).tz_convert("UTC")


def unavailable_day(decision_time, reason):
    first = (utc(decision_time).tz_convert(SG).normalize() + pd.Timedelta(days=1)).tz_convert("UTC")
    times = pd.date_range(first, periods=48, freq="30min")
    return pd.DataFrame({"target_time": times, "target_end": times + pd.Timedelta(minutes=30),
                         "variable": "temperature_c", "location_id": "singapore", "model": "unavailable",
                         "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                         "decision_time": utc(decision_time), "status": "unavailable", "reason": reason, "interpolated": False})


def get_session(archive: Archive, config: dict, model_path=None, now=None, new_session=False) -> dict:
    """Freeze code/config/model identity. A changed champion needs a new experiment."""
    now = utc(now)
    state_path = archive.root / "forward" / "session.json"
    with exclusive_lock(archive.root / ".session.lock"):
        config_hash = content_hash(json_bytes(config))
        current_code = code_version()
        if state_path.exists() and not new_session:
            state = json.loads(state_path.read_text("utf-8"))
            if state["config_hash"] != config_hash:
                raise ValueError("Forward config is frozen; use --new-session for a different experiment")
            if state["code_version"] != current_code:
                raise ValueError("Forward code changed; use --new-session to avoid mixing implementations")
            if model_path is not None and content_hash(Path(model_path).read_bytes()) != state.get("model_checksum"):
                raise ValueError("Champion is frozen; use --new-session to change models")
            return state
        sid = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        directory = archive.root / "forward" / sid
        directory.mkdir(parents=True)
        model_checksum = None
        frozen_model = None
        if model_path is not None:
            payload = Path(model_path).read_bytes()
            model_checksum = content_hash(payload)
            frozen_model = str((directory / "champion.joblib").relative_to(archive.root))
            atomic_write(archive.root / frozen_model, payload)
        state = {"session_id": sid, "started_at": str(now), "start_day": str(now.tz_convert(SG).date()),
                 "config": config, "config_hash": config_hash, "model_checksum": model_checksum,
                 "frozen_model": frozen_model, "code_version": current_code,
                 "shakeout_days": int(config.get("shakeout_days", 14)), "evaluation_days": int(config.get("evaluation_days", 90))}
        atomic_write(directory / "session.json", json_bytes(state))
        atomic_write(state_path, json_bytes(state))
        return state


def run_forward(archive: Archive, config: dict, model_path=None, now=None, new_session=False) -> list[dict]:
    """Record every elapsed scheduled day; never manufacture a missed prediction.

    The optional clock is for deterministic tests only; the CLI always uses wall time.
    """
    from .modeling import predict_day

    injected_clock = now is not None
    started = utc(now)
    session = get_session(archive, config, model_path, started, new_session)
    records = archive.read()
    if session["frozen_model"] and content_hash((archive.root / session["frozen_model"]).read_bytes()) != session["model_checksum"]:
        raise ValueError("Frozen model checksum mismatch; refusing modified champion")
    bundle = joblib.load(archive.root / session["frozen_model"]) if session["frozen_model"] else None
    first_day = pd.Timestamp(session["start_day"])
    today = pd.Timestamp(started.tz_convert(SG).date())
    last_day = min(today, first_day + pd.Timedelta(days=session["shakeout_days"] + session["evaluation_days"] - 1))
    results = []
    for day in pd.date_range(first_day, last_day):
        deadline = daily_deadline(day)
        freeze = deadline - pd.Timedelta(minutes=config.get("freeze_minutes_before_noon", 5))
        run_id = f"forward-{session['session_id']}-{day.date()}"
        existing = archive.root / "runs" / run_id / "manifest.json"
        if existing.exists():
            continue
        if started < freeze:
            continue
        offset = (day - first_day).days
        phase = "shakeout" if offset < session["shakeout_days"] else "evaluation"
        snapshot = records.iloc[0:0].copy()
        error = None
        published = started
        if started > deadline:
            error = "missed_deadline"
            predictions = unavailable_day(deadline, error)
        else:
            # Even published historical facts arriving after the freeze cannot enter live decisions.
            locally_known = records.loc[pd.to_datetime(records.ingested_at, utc=True).le(freeze)] if not records.empty else records
            snapshot = point_in_time(locally_known, freeze, mode="strict")
            try:
                predictions = predict_day(snapshot, deadline, bundle=bundle, config={**config, "mode": "strict"})
                if predictions.empty or predictions.status.eq("unavailable").all():
                    error = "no_usable_inputs"
            except Exception as exc:
                error = f"prediction_failed:{type(exc).__name__}:{exc}"
                predictions = unavailable_day(deadline, error)
            published = started if injected_clock else utc()
            if published > deadline:
                error = "publication_after_deadline"
                predictions = unavailable_day(deadline, error)
        manifest = {"kind": "forward", "session_id": session["session_id"], "phase": phase, "mode": "strict",
                    "decision_time": str(deadline), "freeze_at": str(freeze), "published_at": str(published),
                    "timely": published <= deadline and error is None, "error": error, "config": config,
                    "model_checksum": session["model_checksum"], "frozen_model": session["frozen_model"],
                    "snapshot": archive.snapshot(snapshot), "clock": "test" if injected_clock else "wall"}
        try:
            archive.save_run(run_id, predictions, manifest, snapshot=snapshot)
        except FileExistsError:
            # Another scheduled invocation won. The first persisted decision is authoritative.
            continue
        results.append({"run_id": run_id, "timely": manifest["timely"], "phase": phase, "error": error})
    return results


def replay_forward(archive: Archive, run_id: str) -> pd.DataFrame:
    from .modeling import predict_day
    directory = archive.root / "runs" / run_id
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    if manifest.get("kind") != "forward":
        raise ValueError("Replay requires a forward run")
    if manifest.get("error"):
        return pd.read_parquet(directory / "predictions.parquet")
    if manifest["code_version"] != code_version():
        raise ValueError("Replay requires the recorded implementation version")
    model = archive.root / manifest["frozen_model"] if manifest.get("frozen_model") else None
    if model is not None and content_hash(model.read_bytes()) != manifest["model_checksum"]:
        raise ValueError("Frozen model checksum mismatch")
    snapshot = pd.read_parquet(directory / "snapshot.parquet")
    if archive.snapshot(snapshot)["snapshot_hash"] != manifest["snapshot"]["snapshot_hash"]:
        raise ValueError("Snapshot checksum mismatch")
    return predict_day(snapshot, manifest["decision_time"], bundle=joblib.load(model) if model else None,
                       config={**manifest["config"], "mode": "strict"})


def score_forward(archive: Archive, run_id: str, as_of=None) -> Path:
    from .evaluation import score_predictions
    as_of = utc(as_of)
    directory = archive.root / "runs" / run_id
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    if manifest.get("kind") != "forward":
        raise ValueError("Scoring requires a forward run")
    records = archive.read()
    predictions = pd.read_parquet(directory / "predictions.parquet")
    scores = score_predictions(predictions, records, as_of=as_of)
    version = as_of.strftime("%Y%m%dT%H%M%S%f")
    target = directory / "verification" / version
    target.mkdir(parents=True, exist_ok=False)
    scores.to_parquet(target / "scores.parquet", index=False)
    known = point_in_time(records, as_of)
    atomic_write(target / "manifest.json", json_bytes({"as_of": str(as_of), "snapshot": archive.snapshot(known),
                                                       "source_run": run_id, "code_version": code_version()}))
    return target
