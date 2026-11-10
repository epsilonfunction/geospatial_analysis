import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from weather_engine import sources as s


def test_live_nea_preserves_bytes_and_converts_wind(monkeypatch):
    body = {"code": 0, "data": {"stations": [{"id": "S50", "location": {"latitude": 1.3, "longitude": 103.8}}], "readingUnit": "knots", "readings": [{"timestamp": "2026-10-05T14:00:00+08:00", "data": [{"stationId": "S50", "value": 10}]}]}}
    payload = json.dumps(body).encode()
    monkeypatch.setattr(s._Http, "get", lambda self, url, **kwargs: payload)
    now = "2026-10-05T06:05:00Z"
    batch = s.fetch_nea("wind-speed", now=now)
    record = batch.records.iloc[0]
    assert batch.payload == payload
    assert record.value == pytest.approx(5.144444444)
    assert record.available_at == pd.Timestamp(now)
    assert record.valid_time == pd.Timestamp("2026-10-05T06:00:00Z")
    assert record.availability_confidence == "observed"
    assert set(s.RECORD_COLUMNS) <= set(batch.records.columns)


def test_historical_nea_updates_are_not_publication_evidence(tmp_path):
    path = tmp_path / "rain.csv"
    path.write_text("timestamp,update_timestamp,station_id,location_latitude,location_longitude,reading_value,reading_unit\n2024-01-01T00:05:00+08:00,2024-01-01T00:09:00+08:00,S1,1.3,103.8,2.5,mm\n2024-01-01T00:10:00+08:00,2024-01-01T00:14:00+08:00,S1,1.3,103.8,,mm\n")
    batch = s.parse_nea_csv(path, "rainfall")
    record = batch.records.iloc[0]
    assert pd.isna(record.available_at)
    assert record.availability_confidence == "unknown"
    assert record.valid_time == pd.Timestamp("2023-12-31T16:00:00Z")
    assert record.valid_end - record.valid_time == pd.Timedelta(minutes=5)
    assert batch.metadata["excluded_records"] == 1
    known = s.parse_nea_csv(path, "rain_mm", available_at="2025-01-01T00:00:00Z")
    assert known.records.iloc[0].availability_confidence == "published"


def test_demand_half_hours_and_publication_guard(tmp_path):
    path = tmp_path / "demand.csv"
    path.write_text("Date,Period,Demand (MW)\n2024-01-01,1,6200\n2024-01-01,48,5900\n")
    batch = s.parse_demand_csv(path, "2024-02-01T00:00:00Z")
    assert batch.records.valid_time.iloc[0] == pd.Timestamp("2023-12-31T16:00:00Z")
    assert batch.records.valid_end.iloc[-1] == pd.Timestamp("2024-01-01T16:00:00Z")
    assert batch.records.value.tolist() == [6200, 5900]
    assert batch.records.kind.eq("demand").all()
    with pytest.raises(ValueError, match="cannot precede"):
        s.parse_demand_csv(path, "2023-12-31T16:00:00Z")
    with pytest.raises(ValueError, match="Only explicitly"):
        s.parse_demand_csv(path, None, semantics="demand_forecast")
    assert s.parse_demand_csv(path, None).records.available_at.isna().all()


def test_demand_rejects_noninteger_period(tmp_path):
    path = tmp_path / "demand.csv"
    path.write_text("Date,Period,Demand (MW)\n2024-01-01,1.5,6200\n")
    with pytest.raises(ValueError, match="integers"):
        s.parse_demand_csv(path, "2024-02-01T00:00:00Z")


def _raw(variable, start, end, value, unit, step_type="accum", member="0"):
    return dict(variable=variable, location_id="grid:1:104", latitude=1.0, longitude=104.0, init_time=pd.Timestamp("2024-01-01T00:00:00Z"), member=member, value=value, start=start, end=end, step_type=step_type, raw_unit=unit)


def _normalize(rows):
    return s._normalize_grib_rows(rows, source="ecmwf", model="ifs", now=pd.Timestamp("2026-01-01T00:00:00Z"), publication=pd.NaT, confidence="unknown")


def test_precipitation_accumulations_and_explicit_reset():
    rows, dropped = _normalize([_raw("rain_mm", 0, 3, .001, "m"), _raw("rain_mm", 0, 6, .004, "m"), _raw("rain_mm", 6, 9, .002, "m")])
    assert dropped == 0
    assert [r["value"] for r in rows] == pytest.approx([1, 3, 2])
    assert [r["valid_time"].hour for r in rows] == [0, 3, 6]
    assert [r["valid_end"].hour for r in rows] == [3, 6, 9]
    assert all(pd.isna(r["available_at"]) for r in rows)


def test_cumulative_radiation_energy_and_average_flux():
    accumulated, _ = _normalize([_raw("shortwave_wm2", 0, 3, 1080000, "J m**-2"), _raw("shortwave_wm2", 0, 6, 4320000, "J m**-2")])
    averaged, _ = _normalize([_raw("shortwave_wm2", 0, 3, 100, "W m**-2", "avg"), _raw("shortwave_wm2", 0, 6, 200, "W m**-2", "avg")])
    assert [r["value"] for r in accumulated] == pytest.approx([100, 300])
    assert [r["value"] for r in averaged] == pytest.approx([100, 300])
    assert [r["valid_time"].hour for r in averaged] == [0, 3]


def test_ambiguous_negative_accumulation_is_excluded():
    rows, dropped = _normalize([_raw("rain_mm", 0, 3, 4, "kg m**-2"), _raw("rain_mm", 0, 6, 1, "kg m**-2")])
    assert dropped == 1
    assert len(rows) == 1


def test_units_and_vector_wind():
    rows, _ = _normalize([_raw("temperature_c", 0, 3, 300, "K", "instant"), _raw("cloud_pct", 0, 3, .4, "(0 - 1)", "instant"), _raw("wind_u_ms", 0, 3, 3, "m s**-1", "instant"), _raw("wind_v_ms", 0, 3, 4, "m s**-1", "instant")])
    values = {r["variable"]: r["value"] for r in rows}
    assert values["temperature_c"] == pytest.approx(26.85)
    assert values["cloud_pct"] == 40
    assert values["wind_speed_ms"] == 5


def test_noaa_index_ranges_keep_only_surface_fields():
    index = b"1:0:d=2024010100:TMP:surface:3 hour fcst:\n2:100:d=2024010100:TMP:2 m above ground:3 hour fcst:\n3:300:d=2024010100:UGRD:10 m above ground:3 hour fcst:\n4:450:d=2024010100:TMP:500 mb:3 hour fcst:\n"
    assert s._noaa_ranges(index) == [(100, 299), (300, 449)]


def test_ecmwf_index_selects_member():
    entries = [{"_offset": 0, "_length": 100, "param": "2t", "number": "1", "levtype": "sfc"}, {"_offset": 100, "_length": 150, "param": "2t", "number": "2", "levtype": "sfc"}]
    assert s._ecmwf_ranges("\n".join(map(json.dumps, entries)).encode(), [2]) == [(100, 249)]


class Response:
    def __init__(self, payload=b"GRIB7777", status=206, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {"Content-Range": "bytes 0-7/8"}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)
    def iter_content(self, chunk_size):
        yield self.payload
    def close(self):
        pass


def test_http_retries_and_refuses_full_file(monkeypatch):
    replies = [Response(status=503), Response()]
    monkeypatch.setattr(s.requests, "get", lambda *a, **k: replies.pop(0))
    http = s._Http(s.HttpConfig(attempts=2, backoff_seconds=0))
    assert http.get("https://test/file", headers={"Range": "bytes=0-7"}) == b"GRIB7777"
    assert http.requests == 2
    monkeypatch.setattr(s.requests, "get", lambda *a, **k: Response(status=200))
    with pytest.raises(ValueError, match="ignored"):
        s._Http().get("https://test/file", headers={"Range": "bytes=0-7"})


def test_http_byte_budget(monkeypatch):
    monkeypatch.setattr(s.requests, "get", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="byte budget"):
        s._Http(s.HttpConfig(max_bytes=4)).get("https://test/file")


def test_partial_ensemble_metadata(monkeypatch):
    index = b"1:0:d=2024010100:TMP:2 m above ground:3 hour fcst:\n"
    monkeypatch.setattr(s._Http, "get", lambda self, url, **kwargs: index if url.endswith(".idx") else b"GRIB7777")
    def parse(path, source, model, **kwargs):
        rows, _ = _normalize([_raw("temperature_c", 0, 3, 300, "K", "instant")])
        return s._batch(path.read_bytes(), path.name, {}, rows)
    monkeypatch.setattr(s, "parse_grib", parse)
    batches = s.fetch_noaa("gefs", "2024-01-01T00:00:00Z", [3], [0], now="2024-01-01T06:00:00Z")
    assert not batches[0].metadata["ensemble_complete"]
    assert batches[0].metadata["expected_ensemble_members"] == 31
    assert batches[0].metadata["requested_members"] == ["0"]


def test_grib_decode_crop_and_unknown_availability(tmp_path, monkeypatch):
    import cfgrib
    import xarray as xr
    init = np.datetime64("2024-01-01T00:00:00")
    dataset = xr.Dataset({"t2m": (("latitude", "longitude"), np.full((2, 2), 300.0))}, coords={"latitude": [1.0, 5.0], "longitude": [104.0, 110.0], "time": init, "step": np.timedelta64(3, "h")})
    dataset.t2m.attrs["GRIB_shortName"] = "2t"
    details = {("2t", pd.Timestamp("2024-01-01T00:00:00Z").value, 3.0, "0"): {"start": 0, "end": 3, "step_type": "instant", "unit": "K"}}
    monkeypatch.setattr(s, "_grib_inventory", lambda path: details)
    monkeypatch.setattr(cfgrib, "open_datasets", lambda *a, **k: [dataset])
    path = tmp_path / "archive.grib2"
    path.write_bytes(b"GRIB7777")
    batch = s.parse_grib(path, "ncar", "gfs")
    assert len(batch.records) == 1
    assert batch.records.iloc[0].value == pytest.approx(26.85)
    assert batch.records.available_at.isna().all()
    assert batch.records.latitude.iloc[0] == 1
