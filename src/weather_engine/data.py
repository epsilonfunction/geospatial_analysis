"""Canonical records, explicit information clocks, and immutable local storage."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

PROCESSING_VERSION = "1"
TIME_COLUMNS = ["init_time", "valid_time", "valid_end", "available_at", "first_seen_at", "ingested_at"]
KEY_COLUMNS = ["kind", "source", "model", "variable", "location_id", "init_time", "valid_time", "valid_end", "member"]
REQUIRED = ["kind", "source", "model", "variable", "location_id", "valid_time", "value", "unit"]
DEFAULTS = {
    "model_version": "unknown", "latitude": np.nan, "longitude": np.nan,
    "init_time": pd.NaT, "valid_end": pd.NaT, "member": "0",
    "available_at": pd.NaT, "availability_confidence": "unknown",
    "first_seen_at": pd.NaT, "ingested_at": pd.NaT, "interpolated": False,
    "processing_version": PROCESSING_VERSION, "raw_checksum": "",
}


def utc(value=None) -> pd.Timestamp:
    result = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("An explicit timezone is required (for example 2024-01-01T04:00:00Z)")
    return result.tz_convert("UTC")


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=_json_default, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def content_hash(value: bytes) -> str:
    return sha256(value).hexdigest()


def normalize_records(frame: pd.DataFrame, raw_checksum: str = "") -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        return pd.DataFrame(columns=[*REQUIRED, *[c for c in DEFAULTS if c not in REQUIRED], "record_id"])
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing record columns: {sorted(missing)}")
    for name, default in DEFAULTS.items():
        if name not in frame:
            frame[name] = default
    for name in TIME_COLUMNS:
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    if frame["valid_time"].isna().any():
        raise ValueError("Every record needs a valid_time")
    frame["valid_end"] = frame["valid_end"].fillna(frame["valid_time"])
    if (frame.valid_end < frame.valid_time).any():
        raise ValueError("valid_end cannot precede valid_time")
    if not frame.kind.isin(["forecast", "observation", "demand"]).all():
        raise ValueError("Unknown record kind")
    if not frame.availability_confidence.isin(["observed", "published", "reconstructed", "unknown"]).all():
        raise ValueError("Unknown availability confidence")
    verified = frame.availability_confidence.isin(["observed", "published", "reconstructed"])
    if frame.loc[verified, "available_at"].isna().any():
        raise ValueError("Known availability confidence requires available_at")
    fc = frame.kind.eq("forecast")
    if frame.loc[fc, "init_time"].isna().any() or (frame.loc[fc, "init_time"] > frame.loc[fc, "valid_time"]).any():
        raise ValueError("Forecasts require initialization no later than valid_time")
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    if not np.isfinite(frame.value).all():
        raise ValueError("Missing/non-finite values must be excluded with coverage reported")
    for name in ["source", "model", "model_version", "variable", "location_id", "member", "unit"]:
        frame[name] = frame[name].astype(str)
    if raw_checksum:
        frame["raw_checksum"] = raw_checksum
    # Arrival clocks are deliberately excluded: repeated downloads are not new revisions.
    identity = [*KEY_COLUMNS, "model_version", "latitude", "longitude", "value", "unit", "interpolated", "available_at", "availability_confidence", "raw_checksum", "processing_version"]
    frame["record_id"] = [content_hash(json_bytes(row)) for row in frame[identity].astype(str).to_dict("records")]
    return frame.drop_duplicates("record_id").reset_index(drop=True)


def point_in_time(records: pd.DataFrame, as_of, mode="strict", latency_hours=None) -> pd.DataFrame:
    """Return the latest eligible revision of each fact, without backdating arrivals.

    Reconstructed availability is explicitly opt-in. It never converts a known late
    observation into an earlier one. Source-specific latency can be a mapping.
    """
    cutoff = utc(as_of)
    if mode not in {"strict", "reconstructed"}:
        raise ValueError("mode must be strict or reconstructed")
    if records.empty:
        return normalize_records(records)
    frame = normalize_records(records)
    if mode == "reconstructed":
        unknown = frame.availability_confidence.eq("unknown") | frame.available_at.isna()
        if unknown.any():
            if latency_hours is None:
                raise ValueError("Reconstructed replay requires explicit latency_hours")
            if isinstance(latency_hours, dict):
                delay = frame.source.map(latency_hours)
                if delay[unknown].isna().any():
                    raise ValueError("Latency must be specified for every unknown source")
            else:
                delay = pd.Series(float(latency_hours), index=frame.index)
            if (delay[unknown] < 0).any():
                raise ValueError("Latency cannot be negative")
            origin = frame.init_time.where(frame.kind.eq("forecast"), frame.valid_end)
            frame.loc[unknown, "available_at"] = origin[unknown] + pd.to_timedelta(delay[unknown], unit="h")
            frame.loc[unknown, "availability_confidence"] = "reconstructed"
        allowed = ["observed", "published", "reconstructed"]
    else:
        allowed = ["observed", "published"]
    frame = frame.loc[frame.availability_confidence.isin(allowed) & frame.available_at.le(cutoff)]
    frame = frame.loc[~frame.kind.eq("forecast") | frame.init_time.le(cutoff)]
    # An observation of a future interval cannot be a feature, even with bad metadata.
    frame = frame.loc[frame.kind.eq("forecast") | frame.valid_end.le(cutoff)]
    return (frame.sort_values(["available_at", "first_seen_at", "record_id"], na_position="first")
            .drop_duplicates(KEY_COLUMNS, keep="last").reset_index(drop=True))


def code_version() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
        head = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = "unversioned"
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return f"{head}:{digest.hexdigest()}"


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another writer holds {path}; after a crash, verify no writer is running before removing the lock") from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def atomic_write(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Archive:
    def __init__(self, root="data/weather_engine"):
        self.root = Path(root)

    def ingest(self, batch, now=None) -> dict:
        """Commit bytes and derived rows, deduplicated by payload + parser metadata."""
        now = utc(now)
        raw_hash = content_hash(batch.payload)
        metadata = dict(batch.metadata)
        normalized = normalize_records(batch.records, raw_hash)
        # Interpretation metadata is part of identity; transient download clocks are not.
        stable_meta = {k: v for k, v in metadata.items() if k not in {"fetched_at", "ingested_at", "first_seen_at", "available_at"}}
        # Include known historical publication evidence, but not each live fetch's clock.
        if not normalized.empty and not normalized.availability_confidence.eq("observed").all():
            stable_meta["availability"] = sorted(normalized.available_at.astype(str).unique())
        batch_id = content_hash(raw_hash.encode() + json_bytes(stable_meta) + PROCESSING_VERSION.encode())
        manifest_path = self.root / "ingestions" / f"{batch_id}.json"
        with exclusive_lock(self.root / ".archive.lock"):
            if manifest_path.exists():
                result = json.loads(manifest_path.read_text("utf-8"))
                return {**result, "duplicate": True}
            raw_path = self.root / "raw" / raw_hash[:2] / raw_hash
            if not raw_path.exists():
                atomic_write(raw_path, batch.payload)
            frame = normalized
            if not frame.empty:
                frame["first_seen_at"] = frame.first_seen_at.fillna(now)
                frame["ingested_at"] = now
            record_path = self.root / "records" / f"{batch_id}.parquet"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = record_path.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, record_path)
            result = {"batch_id": batch_id, "raw_checksum": raw_hash, "raw_path": str(raw_path.relative_to(self.root)),
                      "records_path": str(record_path.relative_to(self.root)), "filename": batch.filename,
                      "metadata": metadata, "rows": len(frame), "ingested_at": str(now),
                      "processing_version": PROCESSING_VERSION, "duplicate": False}
            atomic_write(manifest_path, json_bytes(result))
            return result

    def read(self) -> pd.DataFrame:
        # Only committed manifest paths are visible; interrupted writes remain invisible.
        paths = [str(self.root / json.loads(p.read_text("utf-8"))["records_path"])
                 for p in sorted((self.root / "ingestions").glob("*.json"))]
        if not paths:
            return normalize_records(pd.DataFrame())
        import duckdb
        with duckdb.connect() as connection:
            frame = connection.read_parquet(paths, union_by_name=True).df()
        return normalize_records(frame).sort_values("ingested_at").drop_duplicates("record_id", keep="first")

    def snapshot(self, records: pd.DataFrame) -> dict:
        # Recompute identities rather than trusting record_id columns in an artifact.
        normalized = normalize_records(records)
        ids = sorted(normalized.record_id.tolist()) if "record_id" in normalized else []
        checksums = sorted(records.raw_checksum.dropna().unique()) if "raw_checksum" in records else []
        canonical = normalized.sort_values("record_id").reindex(sorted(normalized.columns), axis=1).astype(str).to_dict("records")
        return {"record_ids": ids, "raw_checksums": checksums, "snapshot_hash": content_hash(json_bytes(canonical)), "rows": len(records)}

    def save_run(self, run_id: str, predictions: pd.DataFrame, manifest: dict, scores=None, snapshot=None) -> Path:
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a single safe path component")
        target = self.root / "runs" / run_id
        with exclusive_lock(self.root / ".runs.lock"):
            if target.exists():
                raise FileExistsError(f"Run {run_id} is immutable")
            temp = self.root / "runs" / f".{run_id}.{uuid4().hex}"
            temp.mkdir(parents=True)
            predictions.to_parquet(temp / "predictions.parquet", index=False)
            if isinstance(scores, pd.DataFrame):
                scores.to_parquet(temp / "scores.parquet", index=False)
            if snapshot is not None:
                snapshot.to_parquet(temp / "snapshot.parquet", index=False)
            atomic_write(temp / "manifest.json", json_bytes({"run_id": run_id, "code_version": code_version(), **manifest}))
            temp.rename(target)
        return target
