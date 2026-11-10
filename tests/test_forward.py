from types import SimpleNamespace

import pandas as pd
import pytest

from weather_engine.data import Archive
from weather_engine.forward import get_session, replay_forward, run_forward, unavailable_day


def test_missed_deadline_records_failure_without_hindsight(tmp_path):
    archive = Archive(tmp_path)
    result = run_forward(archive, {}, now="2024-01-01T04:01:00Z")
    assert result[0]["error"] == "missed_deadline"
    assert not result[0]["timely"]
    assert run_forward(archive, {}, now="2024-01-01T06:00:00Z") == []
    assert replay_forward(archive, result[0]["run_id"]).status.eq("unavailable").all()


def test_forward_uses_frozen_inputs_and_replays(monkeypatch, tmp_path):
    from weather_engine import modeling
    archive = Archive(tmp_path)
    rows = pd.DataFrame([{"kind": "observation", "source": "nea", "model": "station", "variable": "temperature_c",
                          "location_id": "S1", "valid_time": "2024-01-01T01:00:00Z", "value": 28, "unit": "degC",
                          "available_at": "2024-01-01T01:01:00Z", "availability_confidence": "published"}])
    archive.ingest(SimpleNamespace(payload=b"known", filename="fixture", metadata={}, records=rows), now="2024-01-01T01:02:00Z")
    rows.loc[0, "value"] = 99
    rows.loc[0, "available_at"] = "2024-01-01T03:58:00Z"
    archive.ingest(SimpleNamespace(payload=b"late", filename="fixture", metadata={}, records=rows), now="2024-01-01T03:58:00Z")
    def predict(records, decision_time, **kwargs):
        output = unavailable_day(decision_time, "fixture")
        output["mean"] = records.value.mean()
        output["status"] = "forecast"
        return output
    monkeypatch.setattr(modeling, "predict_day", predict)
    result = run_forward(archive, {}, now="2024-01-01T03:59:00Z")
    assert result[0]["timely"]
    saved = pd.read_parquet(tmp_path / "runs" / result[0]["run_id"] / "predictions.parquet")
    assert saved["mean"].eq(28).all()
    pd.testing.assert_frame_equal(saved, replay_forward(archive, result[0]["run_id"]))


def test_session_freezes_config_and_champion(tmp_path):
    archive = Archive(tmp_path)
    get_session(archive, {"a": 1}, now="2024-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="frozen"):
        get_session(archive, {"a": 2}, now="2024-01-02T00:00:00Z")
    assert get_session(archive, {"a": 2}, now="2024-01-02T00:00:00Z", new_session=True)["config"] == {"a": 2}


def test_backlog_marks_every_elapsed_day(tmp_path):
    archive = Archive(tmp_path)
    assert run_forward(archive, {}, now="2024-01-01T00:00:00Z") == []
    result = run_forward(archive, {}, now="2024-01-03T05:00:00Z")
    assert len(result) == 3
    assert all(r["error"] == "missed_deadline" for r in result)
