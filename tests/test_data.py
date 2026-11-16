from types import SimpleNamespace

import pandas as pd
import pytest

from weather_engine.data import Archive, normalize_records, point_in_time


def fact(value=28, available="2024-01-01T01:00:00Z", **extra):
    return {"kind": "observation", "source": "nea", "model": "station", "variable": "temperature_c",
            "location_id": "S1", "valid_time": "2024-01-01T00:00:00Z", "valid_end": "2024-01-01T00:30:00Z",
            "value": value, "unit": "degC", "available_at": available,
            "availability_confidence": "published" if available else "unknown", **extra}


def test_point_in_time_excludes_future_revision_and_future_observation():
    rows = [fact(), fact(99, "2024-01-02T00:00:00Z"),
            fact(31, valid_time="2024-01-02T00:00:00Z", valid_end="2024-01-02T00:30:00Z")]
    result = point_in_time(pd.DataFrame(rows), "2024-01-01T04:00:00Z")
    assert result.value.tolist() == [28]


def test_unknown_is_never_treated_as_historical_publication():
    records = pd.DataFrame([fact(available=None)])
    assert point_in_time(records, "2024-01-01T04:00:00Z").empty
    with pytest.raises(ValueError, match="latency"):
        point_in_time(records, "2024-01-01T04:00:00Z", "reconstructed")
    result = point_in_time(records, "2024-01-01T04:00:00Z", "reconstructed", 2)
    assert result.availability_confidence.tolist() == ["reconstructed"]
    assert result.available_at.iloc[0] == pd.Timestamp("2024-01-01T02:30:00Z")


def test_reconstruction_never_backdates_known_late_arrival():
    records = pd.DataFrame([fact(available="2024-02-01T00:00:00Z", availability_confidence="observed")])
    assert point_in_time(records, "2024-01-01T04:00:00Z", "reconstructed", 0).empty


def test_bad_publication_clock_cannot_admit_future_model_cycle():
    rows = pd.DataFrame([fact(kind="forecast", source="noaa", model="gfs", init_time="2024-01-02T00:00:00Z",
                              valid_time="2024-01-02T03:00:00Z", valid_end="2024-01-02T03:00:00Z")])
    assert point_in_time(rows, "2024-01-01T04:00:00Z").empty


def test_archive_idempotency_and_revision_lineage(tmp_path):
    archive = Archive(tmp_path)
    rows = pd.DataFrame([fact()])
    batch = SimpleNamespace(payload=b"original", filename="nea.json", records=rows, metadata={"source": "nea"})
    one = archive.ingest(batch, now="2024-01-01T01:01:00Z")
    two = archive.ingest(batch, now="2024-01-02T01:01:00Z")
    assert one["batch_id"] == two["batch_id"]
    assert two["duplicate"]
    revised = SimpleNamespace(payload=b"corrected", filename="nea.json", records=pd.DataFrame([fact(29, "2024-01-03T00:00:00Z")]), metadata={"source": "nea"})
    archive.ingest(revised, now="2024-01-03T00:01:00Z")
    records = archive.read()
    assert len(records) == 2
    assert records.raw_checksum.nunique() == 2
    assert point_in_time(records, "2024-01-02T04:00:00Z").value.tolist() == [28]
    assert point_in_time(records, "2024-01-04T04:00:00Z").value.tolist() == [29]


def test_live_duplicate_preserves_original_availability(tmp_path):
    archive = Archive(tmp_path)
    for day in [1, 2]:
        rows = pd.DataFrame([fact(available=f"2024-01-0{day}T01:00:00Z", availability_confidence="observed")])
        batch = SimpleNamespace(payload=b"unchanged", filename="x", records=rows, metadata={"source": "nea", "fetched_at": str(day)})
        archive.ingest(batch, now=f"2024-01-0{day}T01:01:00Z")
    records = archive.read()
    assert len(records) == 1
    assert records.available_at.iloc[0] == pd.Timestamp("2024-01-01T01:00:00Z")


def test_snapshot_hash_detects_value_tampering(tmp_path):
    archive = Archive(tmp_path)
    records = normalize_records(pd.DataFrame([fact()]))
    before = archive.snapshot(records)
    records.loc[0, "value"] = 90
    assert before["snapshot_hash"] != archive.snapshot(records)["snapshot_hash"]


def test_run_cannot_be_overwritten(tmp_path):
    archive = Archive(tmp_path)
    archive.save_run("demo", pd.DataFrame({"value": [1]}), {"kind": "test"})
    with pytest.raises(FileExistsError):
        archive.save_run("demo", pd.DataFrame({"value": [2]}), {})
    with pytest.raises(ValueError):
        archive.save_run("../escape", pd.DataFrame(), {})
