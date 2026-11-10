"""Bounded public-data adapters and explicit-vintage weather normalization.

Downloads return immutable bytes alongside derived records. No adapter treats a
forecast initialization timestamp as its publication timestamp. Historical files
need explicit publication evidence to participate in strict historical replay.

HTTP limits are configured with WEATHER_HTTP_ATTEMPTS, CONNECT_TIMEOUT,
READ_TIMEOUT, MAX_REQUESTS, MAX_BYTES, MAX_FILES and BACKOFF_SECONDS (each name
has the WEATHER_HTTP_ prefix). Defaults permit pilot/control collections; full
ensembles require an explicit larger transfer budget. Range-selected GRIB
messages still cover the global field; geographic cropping occurs after decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


PROCESSING_VERSION = "sources-1"
BOUNDS = (0.0, 3.0, 102.0, 105.0)  # south, north, west, east
TIME_COLUMNS = ["init_time", "valid_time", "valid_end", "available_at", "first_seen_at", "ingested_at"]
RECORD_COLUMNS = ["kind", "source", "model", "model_version", "variable", "location_id", "latitude", "longitude", "init_time", "valid_time", "valid_end", "member", "value", "unit", "available_at", "availability_confidence", "first_seen_at", "ingested_at", "interpolated", "processing_version"]
NEA = {
    "temperature_c": ("air-temperature", "degC"),
    "humidity_pct": ("relative-humidity", "%"),
    "rain_mm": ("rainfall", "mm"),
    "wind_speed_ms": ("wind-speed", "m/s"),
    "wind_direction_deg": ("wind-direction", "degree"),
}
NEA_ALIASES = {endpoint: name for name, (endpoint, _) in NEA.items()}
UNITS = {"temperature_c": "degC", "dewpoint_c": "degC", "humidity_pct": "%", "rain_mm": "mm", "cloud_pct": "%", "shortwave_wm2": "W/m2", "wind_u_ms": "m/s", "wind_v_ms": "m/s", "wind_speed_ms": "m/s", "demand_mw": "MW"}
GRIB_FIELDS = {
    "2t": "temperature_c", "t2m": "temperature_c",
    "2d": "dewpoint_c", "d2m": "dewpoint_c",
    "2r": "humidity_pct", "r2": "humidity_pct",
    "tp": "rain_mm", "apcp": "rain_mm",
    "tcc": "cloud_pct", "ssrd": "shortwave_wm2", "dswrf": "shortwave_wm2",
    "10u": "wind_u_ms", "u10": "wind_u_ms",
    "10v": "wind_v_ms", "v10": "wind_v_ms",
    "10si": "wind_speed_ms", "si10": "wind_speed_ms",
}
NOAA_SELECTORS = {
    ("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
    ("RH", "2 m above ground"), ("APCP", "surface"),
    ("TCDC", "entire atmosphere"), ("TCDC", "entire atmosphere (considered as a single layer)"),
    ("DSWRF", "surface"), ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
}


@dataclass
class SourceBatch:
    payload: bytes
    filename: str
    metadata: dict
    records: pd.DataFrame


@dataclass(frozen=True)
class HttpConfig:
    """Bounds apply to one adapter call, including retries and index requests."""
    attempts: int = 3
    connect_timeout: float = 10
    read_timeout: float = 45
    max_requests: int = 1024
    max_bytes: int = 512 * 1024 * 1024
    max_files: int = 128
    backoff_seconds: float = 0.5

    @classmethod
    def from_env(cls):
        values = {}
        for field in cls.__dataclass_fields__:
            raw = os.environ.get(f"WEATHER_HTTP_{field.upper()}")
            if raw is not None:
                values[field] = float(raw) if field in {"connect_timeout", "read_timeout", "backoff_seconds"} else int(raw)
        config = cls(**values)
        if min(config.attempts, config.max_requests, config.max_bytes, config.max_files) < 1 or min(config.connect_timeout, config.read_timeout) <= 0 or config.backoff_seconds < 0:
            raise ValueError("HTTP bounds must be positive; backoff must be nonnegative")
        return config


class _Http:
    def __init__(self, config=None):
        self.config = config or HttpConfig.from_env()
        self.requests = 0
        self.bytes = 0

    def get(self, url, *, headers=None):
        """Require range responses: silently downloading global files is forbidden."""
        config = self.config
        request_headers = {"User-Agent": "weather-engine/1 (public weather analytics)", **(headers or {})}
        for attempt in range(config.attempts):
            self.requests += 1
            if self.requests > config.max_requests:
                raise RuntimeError("HTTP request budget exhausted; request fewer leads/members")
            response = None
            try:
                response = requests.get(url, headers=request_headers, timeout=(config.connect_timeout, config.read_timeout), stream=True)
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                if "Range" in request_headers and response.status_code != 206:
                    raise ValueError("Server ignored the requested byte range; full-file fallback is disabled")
                content = bytearray()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    self.bytes += len(chunk)
                    if self.bytes > config.max_bytes:
                        raise RuntimeError("HTTP byte budget exhausted; request fewer leads/members")
                    content.extend(chunk)
                if "Range" in request_headers:
                    wanted = request_headers["Range"].removeprefix("bytes=")
                    first, last = wanted.split("-", 1)
                    actual = response.headers.get("Content-Range", "")
                    if not actual.startswith(f"bytes {first}-"):
                        raise ValueError("Range response does not match requested start")
                    if last and len(content) != int(last) - int(first) + 1:
                        raise ValueError("Truncated or incorrect range response")
                return bytes(content)
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in {429, 500, 502, 503, 504}:
                    raise
                if attempt + 1 == config.attempts:
                    raise
                time.sleep(min(config.backoff_seconds * 2 ** attempt, 5))
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError("Unreachable HTTP retry state")


def _time(value=None, *, local="UTC", explicit=False):
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is None:
        if explicit:
            raise ValueError("Publication and cycle timestamps require an explicit timezone")
        stamp = stamp.tz_localize(local)
    return stamp.tz_convert("UTC")


def _base(kind, source, model, now, available_at=None, confidence="unknown"):
    return dict(kind=kind, source=source, model=model, model_version="unknown", member="0", init_time=pd.NaT, available_at=available_at if available_at is not None else pd.NaT, availability_confidence=confidence, first_seen_at=now, ingested_at=now, interpolated=False, processing_version=PROCESSING_VERSION)


def _records(rows):
    frame = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    for column in TIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _batch(payload, filename, metadata, rows):
    metadata = {"processing_version": PROCESSING_VERSION, "raw_sha256": sha256(payload).hexdigest(), **metadata}
    return SourceBatch(payload, filename, metadata, rows if isinstance(rows, pd.DataFrame) else _records(rows))


def _nea_value(value, variable, unit):
    result = float(value)
    if variable == "wind_speed_ms":
        if unit.lower().strip() in {"knots", "knot", "kt", "kts"}:
            result *= 0.5144444444444445
        elif unit.lower().strip() not in {"m/s", "ms-1", "m s-1"}:
            raise ValueError(f"Unsupported NEA wind unit {unit!r}")
    if variable in {"rain_mm", "wind_speed_ms"} and result < 0:
        return np.nan
    if variable == "humidity_pct" and not 0 <= result <= 100:
        return np.nan
    return result


def _nea_rows(body, variable, now):
    data = body.get("data")
    if body.get("code", 0) != 0 or not isinstance(data, dict):
        raise ValueError("NEA response has an error or missing data object")
    stations = {str(station["id"]): station for station in data.get("stations", [])}
    unit = data.get("readingUnit", data.get("reading_unit", ""))
    rows, skipped = [], 0
    for item in data.get("readings", []):
        valid = _time(item["timestamp"], local="Asia/Singapore")
        for reading in item.get("data", []):
            station_id = str(reading.get("stationId", reading.get("station_id", "")))
            station = stations.get(station_id)
            if station is None or reading.get("value") is None:
                skipped += 1
                continue
            value = _nea_value(reading["value"], variable, unit)
            if not np.isfinite(value):
                skipped += 1
                continue
            start = valid - pd.Timedelta(minutes=5) if variable == "rain_mm" else valid
            location = station.get("location", {})
            rows.append({**_base("observation", "nea", "station", now, now, "observed"), "variable": variable, "location_id": station_id, "latitude": float(location["latitude"]), "longitude": float(location["longitude"]), "valid_time": start, "valid_end": valid, "value": value, "unit": NEA[variable][1]})
    return rows, skipped


def fetch_nea(variable, now=None) -> SourceBatch:
    """Fetch the latest v2 station snapshot; returned bytes include all station metadata."""
    variable = NEA_ALIASES.get(variable, variable)
    if variable not in NEA:
        raise ValueError(f"Unknown NEA variable: {variable}; choose {sorted(NEA)}")
    endpoint = NEA[variable][0]
    url = f"https://api-open.data.gov.sg/v2/real-time/api/{endpoint}"
    headers = {}
    if os.environ.get("DATA_GOV_SG_API_KEY"):
        headers["x-api-key"] = os.environ["DATA_GOV_SG_API_KEY"]
    payload = _Http().get(url, headers=headers)
    observed = _time(now)
    rows, skipped = _nea_rows(json.loads(payload), variable, observed)
    if not rows:
        raise ValueError("NEA response contains no usable station readings")
    return _batch(payload, f"nea_{endpoint}.json", {"source": "nea", "variable": variable, "url": url, "fetched_at": observed.isoformat(), "excluded_records": skipped, "reading_interval_minutes": 5 if variable == "rain_mm" else 0}, rows)


def _columns(frame):
    return {re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_"): name for name in frame.columns}


def _column(columns, *names, required=True):
    for name in names:
        if name in columns:
            return columns[name]
    if required:
        raise ValueError(f"Missing CSV column; expected one of {names}")
    return None


def parse_nea_csv(path, variable, available_at=None) -> SourceBatch:
    """Read the official annual station CSV. Update timestamps are not publication proof."""
    variable = NEA_ALIASES.get(variable, variable)
    if variable not in NEA:
        raise ValueError(f"Unsupported NEA variable: {variable}")
    path = Path(path)
    payload = path.read_bytes()
    frame = pd.read_csv(BytesIO(payload), dtype={"station_id": str})
    columns = _columns(frame)
    timestamp = _column(columns, "timestamp", "valid_time")
    station = _column(columns, "station_id", "location_id")
    value_col = _column(columns, "reading_value", "value", variable)
    lat = _column(columns, "location_latitude", "latitude")
    lon = _column(columns, "location_longitude", "longitude")
    unit_col = _column(columns, "reading_unit", "unit", required=False)
    now = _time()
    publication = _time(available_at, explicit=True) if available_at is not None else pd.NaT
    rows, skipped = [], 0
    for item in frame.to_dict("records"):
        try:
            valid = _time(item[timestamp], local="Asia/Singapore")
            unit = str(item[unit_col]) if unit_col else NEA[variable][1]
            value = _nea_value(item[value_col], variable, unit)
            latitude, longitude = float(item[lat]), float(item[lon])
            if pd.isna(valid) or not np.isfinite([value, latitude, longitude]).all():
                skipped += 1
                continue
        except (ValueError, TypeError):
            skipped += 1
            continue
        rows.append({**_base("observation", "nea", "station", now, publication, "published" if available_at is not None else "unknown"), "variable": variable, "location_id": str(item[station]), "latitude": latitude, "longitude": longitude, "valid_time": valid - pd.Timedelta(minutes=5) if variable == "rain_mm" else valid, "valid_end": valid, "value": value, "unit": NEA[variable][1]})
    if not rows:
        raise ValueError("Historical NEA CSV contains no usable records")
    return _batch(payload, path.name, {"source": "nea", "variable": variable, "excluded_records": skipped, "historical_update_timestamps_are_not_publication_evidence": True, "timezone_for_naive_times": "Asia/Singapore", "available_at": publication.isoformat() if not pd.isna(publication) else None}, rows)


def parse_demand_csv(path, available_at, semantics="actual_system_demand") -> SourceBatch:
    """Import actual half-hour mean MW, with explicit publication evidence.

    Accepted shapes: Date + Period(1..48) + Demand (MW), or
    valid_time + valid_end + demand_mw. Naive delivery times are Singapore time.
    available_at=None deliberately leaves availability unknown for research.
    """
    if semantics != "actual_system_demand":
        raise ValueError("Only explicitly identified actual_system_demand is supported; forecast/net/gross series must not be relabeled")
    path = Path(path)
    payload = path.read_bytes()
    frame = pd.read_csv(BytesIO(payload))
    columns = _columns(frame)
    value_col = _column(columns, "demand_mw", "actual_demand_mw", "system_demand_mw", "demand", "actual_system_demand_mw")
    start_col = _column(columns, "valid_time", "interval_start", required=False)
    end_col = _column(columns, "valid_end", "interval_end", required=False)
    date_col = _column(columns, "date", "trading_date", required=start_col is None)
    period_col = _column(columns, "period", "half_hour_period", "settlement_period", required=start_col is None)
    publication_col = _column(columns, "available_at", "published_at", required=False)
    if available_at is None and publication_col is None:
        publication = pd.NaT
    else:
        publication = _time(available_at, explicit=True) if available_at is not None else None
    now, rows = _time(), []
    for item in frame.to_dict("records"):
        if start_col:
            start = _time(item[start_col], local="Asia/Singapore")
            end = _time(item[end_col], local="Asia/Singapore") if end_col else start + pd.Timedelta(minutes=30)
        else:
            period = float(item[period_col])
            if not period.is_integer() or not 1 <= period <= 48:
                raise ValueError("Singapore half-hour periods must be integers 1..48")
            if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", str(item[date_col]).strip()):
                raise ValueError("Ambiguous demand dates must be converted to YYYY-MM-DD")
            local_date = pd.Timestamp(item[date_col])
            if local_date.tzinfo is not None or local_date != local_date.normalize():
                raise ValueError("Date must be a timezone-free Singapore calendar date")
            start = _time(local_date, local="Asia/Singapore") + pd.Timedelta(minutes=30 * (int(period) - 1))
            end = start + pd.Timedelta(minutes=30)
        if end - start != pd.Timedelta(minutes=30):
            raise ValueError("Actual demand must represent a half-hour mean, not energy or another interval")
        value = float(str(item[value_col]).replace(",", ""))
        if not np.isfinite(value) or value < 0:
            raise ValueError("Actual system demand must be finite nonnegative MW")
        published = publication if publication is not None else _time(item[publication_col], explicit=True)
        if not pd.isna(published) and published < end:
            raise ValueError("Actual demand publication cannot precede the delivery interval end")
        rows.append({**_base("demand", "ema", "actual_system_demand", now, published, "unknown" if pd.isna(published) else "published"), "variable": "demand_mw", "location_id": "SG", "latitude": 1.3521, "longitude": 103.8198, "valid_time": start, "valid_end": end, "value": value, "unit": "MW"})
    records = _records(rows)
    if records.empty:
        raise ValueError("Demand CSV is empty")
    if records.duplicated(["valid_time", "available_at"]).any():
        raise ValueError("Duplicate demand intervals at the same publication time")
    return _batch(payload, path.name, {"source": "ema", "semantics": semantics, "interval": "half-hour mean MW", "timezone_for_naive_times": "Asia/Singapore", "available_at": str(available_at) if available_at is not None else None}, records)


def _selection(cycle, leads, members, maximum_member):
    cycle = _time(cycle, explicit=True)
    if cycle.hour not in {0, 6, 12, 18} or cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("Forecast cycle must be 00/06/12/18 UTC on the hour")
    leads = list(leads)
    members = list(members)
    if not leads or any(float(lead) != int(lead) or not 0 <= int(lead) <= 72 for lead in leads):
        raise ValueError("Select at least one integer lead between 0 and 72 hours")
    parsed = []
    for member in members:
        name = str(member).lower()
        if name in {"control", "cf", "c00", "deterministic"}:
            name = "0"
        elif name.startswith("p"):
            name = name[1:]
        parsed.append(int(name))
    if not parsed or any(member < 0 or member > maximum_member for member in parsed):
        raise ValueError(f"Members must be 0..{maximum_member}")
    return cycle, sorted(set(int(lead) for lead in leads)), sorted(set(parsed))


def _noaa_ranges(index):
    entries = []
    for line in index.decode("utf-8").splitlines():
        parts = line.split(":")
        if len(parts) >= 6 and parts[1].isdigit():
            entries.append((int(parts[1]), (parts[3], parts[4])))
    if not entries or any(a[0] >= b[0] for a, b in zip(entries, entries[1:])):
        raise ValueError("Invalid or unsorted NOAA GRIB index")
    ranges = []
    for i, (offset, field) in enumerate(entries):
        if field in NOAA_SELECTORS:
            end = entries[i + 1][0] - 1 if i + 1 < len(entries) else None
            ranges.append((offset, end))
    return ranges


def _range_payload(http, url, ranges):
    if not ranges:
        raise ValueError(f"No requested weather fields were found in {url}")
    chunks = []
    for start, end in ranges:
        content = http.get(url, headers={"Range": f"bytes={start}-{end if end is not None else ''}"})
        if not content.startswith(b"GRIB") or not content.endswith(b"7777"):
            raise ValueError("Range response is not a complete GRIB message")
        chunks.append(content)
    return b"".join(chunks)


def _parse_download(payload, filename, *, source, model, cycle, observed, objects, members, leads, expected_members):
    # Parse one combined subset so accumulated fields can be differenced across leads.
    with tempfile.TemporaryDirectory(prefix="weather_grib_") as directory:
        path = Path(directory) / filename
        path.write_bytes(payload)
        batch = parse_grib(path, source, model, init_time=cycle, available_at=observed, availability_confidence="observed")
    batch.records["first_seen_at"] = observed
    batch.records["ingested_at"] = observed
    present = sorted(batch.records.member.unique(), key=int) if not batch.records.empty else []
    coverage = {}
    for (member, valid_end), group in batch.records.groupby(["member", "valid_end"]):
        lead = (valid_end - cycle).total_seconds() / 3600
        coverage[(member, lead)] = set(group.variable)
    required = {"temperature_c", "dewpoint_c", "wind_u_ms", "wind_v_ms", "cloud_pct", "rain_mm", "shortwave_wm2"}
    missing = []
    for member in members:
        for lead in leads:
            expected = required - {"rain_mm", "shortwave_wm2"} if lead == 0 else required
            absent = expected - coverage.get((str(member), lead), set())
            if absent:
                missing.append({"member": str(member), "lead": lead, "variables": sorted(absent)})
    complete = not missing
    batch.metadata.update({"cycle": cycle.isoformat(), "source_objects": objects, "requested_members": [str(member) for member in members], "requested_leads": leads, "present_members": present, "missing_requested_fields": missing, "expected_ensemble_members": expected_members, "complete": complete, "ensemble_complete": complete and set(present) == {str(i) for i in range(expected_members)}, "fetched_at": observed.isoformat()})
    return batch


def _preflight(http, *, files, selected_messages):
    if files > http.config.max_files:
        raise ValueError(f"Selection needs {files} files; WEATHER_HTTP_MAX_FILES={http.config.max_files}. Request fewer leads/members or explicitly raise the budget.")
    estimated_requests = files + selected_messages
    if estimated_requests > http.config.max_requests:
        raise ValueError(f"Selection may need {estimated_requests} index/range requests; WEATHER_HTTP_MAX_REQUESTS={http.config.max_requests}. Request fewer leads/members or explicitly raise the budget; ensembles are never silently reduced.")


def fetch_noaa(model, cycle, leads, members, now=None) -> list[SourceBatch]:
    """Retrieve selected GFS/GEFS messages from NOAA's public cloud indices.

    GEFS uses 0.5-degree pgrb2a fields, which include the required surface fluxes.
    The adapter aborts incomplete downloads; it never silently fills missing runs.
    """
    model = model.lower()
    if model not in {"gfs", "gefs"}:
        raise ValueError("NOAA model must be gfs or gefs")
    cycle, leads, members = _selection(cycle, leads, members, 0 if model == "gfs" else 30)
    if model == "gefs" and any(lead % 3 for lead in leads):
        raise ValueError("GEFS requires 3-hour forecast steps in this adapter")
    http = _Http()
    _preflight(http, files=len(leads) * len(members), selected_messages=len(leads) * len(members) * len(NOAA_SELECTORS))
    date, hour = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    payloads, objects = [], []
    for member in members:
        for lead in leads:
            if model == "gfs":
                url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{lead:03d}"
            else:
                name = "gec00" if member == 0 else f"gep{member:02d}"
                url = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{date}/{hour}/atmos/pgrb2ap5/{name}.t{hour}z.pgrb2a.0p50.f{lead:03d}"
            index = http.get(url + ".idx")
            ranges = _noaa_ranges(index)
            content = _range_payload(http, url, ranges)
            payloads.append(content)
            objects.append({"url": url, "index_sha256": sha256(index).hexdigest(), "subset_sha256": sha256(content).hexdigest(), "ranges": ranges, "lead": lead, "member": str(member)})
    observed = _time(now)
    filename = f"{model}_{date}{hour}_subset.grib2"
    return [_parse_download(b"".join(payloads), filename, source="noaa", model=model, cycle=cycle, observed=observed, objects=objects, members=members, leads=leads, expected_members=1 if model == "gfs" else 31)]


def _ecmwf_ranges(index, members):
    wanted = {"2t", "2d", "10u", "10v", "tp", "tcc", "ssrd"}
    ranges = []
    for line in index.decode("utf-8").splitlines():
        item = json.loads(line)
        if item.get("param") not in wanted or item.get("levtype", "sfc") != "sfc":
            continue
        member = int(item.get("number", 0))
        if member in members:
            start = int(item["_offset"])
            ranges.append((start, start + int(item["_length"]) - 1))
    return ranges


def fetch_ecmwf(cycle, leads, members, now=None) -> list[SourceBatch]:
    """Download the free 0.25-degree IFS control/perturbed forecast subset.

    ECMWF .index records allow byte-range selection without downloading a global
    51-member file. Member 0 uses oper/fc; members 1..50 use enfo/pf.
    """
    cycle, leads, members = _selection(cycle, leads, members, 50)
    if any(lead % 3 for lead in leads):
        raise ValueError("The ECMWF free subset requires 3-hour steps through hour 72")
    http = _Http()
    groups = [("oper", "fc", [0])] if 0 in members else []
    perturbed = [member for member in members if member]
    if perturbed:
        groups.append(("enfo", "pf", perturbed))
    _preflight(http, files=len(leads) * len(groups), selected_messages=len(leads) * len(members) * 7)
    date, hour = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    payloads, objects = [], []
    for stream, forecast_type, selected in groups:
        for lead in leads:
            filename = f"{date}{hour}0000-{lead}h-{stream}-{forecast_type}.grib2"
            url = f"https://data.ecmwf.int/forecasts/{date}/{hour}z/ifs/0p25/{stream}/{filename}"
            index = http.get(url.removesuffix(".grib2") + ".index")
            ranges = _ecmwf_ranges(index, selected)
            content = _range_payload(http, url, ranges)
            payloads.append(content)
            objects.append({"url": url, "index_sha256": sha256(index).hexdigest(), "subset_sha256": sha256(content).hexdigest(), "ranges": ranges, "lead": lead, "members": [str(m) for m in selected]})
    observed = _time(now)
    return [_parse_download(b"".join(payloads), f"ifs_{date}{hour}_subset.grib2", source="ecmwf", model="ifs", cycle=cycle, observed=observed, objects=objects, members=members, leads=leads, expected_members=51)]


def _grib_inventory(path):
    import eccodes
    inventory = {}
    with open(path, "rb") as stream:
        while True:
            handle = eccodes.codes_grib_new_from_file(stream)
            if handle is None:
                break
            try:
                def get(key, default=None):
                    try:
                        return eccodes.codes_get(handle, key)
                    except eccodes.CodesInternalError:
                        return default
                date, clock = get("dataDate"), int(get("dataTime", 0))
                init = _time(f"{date} {clock // 100:02d}:{clock % 100:02d}")
                step_unit = int(get("stepUnits", 1))
                factors = {0: 1 / 60, 1: 1, 2: 24, 10: 3, 11: 6, 12: 12, 13: 1 / 3600}
                if step_unit not in factors:
                    raise ValueError(f"Unsupported GRIB step unit {step_unit}")
                start = float(get("startStep", 0)) * factors[step_unit]
                end = float(get("endStep", 0)) * factors[step_unit]
                key = (str(get("shortName")), init.value, end, str(get("number", 0)))
                inventory[key] = {"start": start, "end": end, "step_type": str(get("stepType", "instant")), "unit": str(get("units", "")), "level": get("level"), "level_type": get("typeOfLevel")}
            finally:
                eccodes.codes_release(handle)
    return inventory


def _unit_value(value, variable, unit):
    unit = unit.lower().replace(" ", "").replace("**", "^")
    if variable in {"temperature_c", "dewpoint_c"}:
        if unit in {"k", "kelvin"}:
            return value - 273.15
        if unit not in {"c", "degc", "celsius", "°c"}:
            raise ValueError(f"Unsupported temperature unit {unit}")
    elif variable == "rain_mm":
        if unit in {"m", "mofwater equivalent", "mofwaterequivalent"}:
            return value * 1000
        if unit not in {"mm", "kgm^-2", "kgm-2", "kg/m^2", "kg/m2"}:
            raise ValueError(f"Unsupported precipitation unit {unit}")
    elif variable in {"cloud_pct", "humidity_pct"}:
        if unit in {"(0-1)", "0-1", "1", "fraction", "proportion"}:
            return value * 100
        if unit not in {"%", "percent", "percentage"}:
            raise ValueError(f"Unsupported percentage unit {unit}")
    elif variable == "shortwave_wm2":
        if unit not in {"jm^-2", "jm-2", "j/m^2", "j/m2", "wm^-2", "wm-2", "w/m^2", "w/m2"}:
            raise ValueError(f"Unsupported radiation unit {unit}")
    elif variable.startswith("wind_") and unit not in {"ms^-1", "ms-1", "m/s"}:
        raise ValueError(f"Unsupported wind unit {unit}")
    return value


def _normalize_grib_rows(raw_rows, *, source, model, now, publication, confidence):
    """Difference cumulative blocks using their actual GRIB start/end steps."""
    groups = {}
    for row in raw_rows:
        key = (row["variable"], row["location_id"], row["init_time"], row["member"])
        groups.setdefault(key, []).append(row)
    output, dropped = [], 0
    for rows in groups.values():
        previous = {}
        for item in sorted(rows, key=lambda row: (row["end"], row["start"])):
            variable = item["variable"]
            value = _unit_value(item["value"], variable, item["raw_unit"])
            start, end = item["start"], item["end"]
            interval = item["step_type"] in {"accum", "avg", "average"}
            if start < 0 or end < start:
                dropped += 1
                continue
            energy = variable == "shortwave_wm2" and item["raw_unit"].lower().lstrip().startswith("j")
            if interval and end > start:
                # Radiation mean blocks are converted to energy before differencing.
                quantity = value * (end - start) * 3600 if variable == "shortwave_wm2" and not energy else value
                before = previous.get(start)
                previous[start] = (end, quantity)
                if before is not None and before[0] < end:
                    value = quantity - before[1]
                    start = before[0]
                else:
                    value = quantity
                if variable == "shortwave_wm2":
                    value /= (end - start) * 3600
            elif energy:
                # Accumulation step zero has no interval and cannot define a flux.
                dropped += 1
                continue
            if variable in {"rain_mm", "shortwave_wm2", "wind_speed_ms"} and value < -1e-6:
                dropped += 1
                continue
            if variable in {"rain_mm", "shortwave_wm2"}:
                value = max(0, value)
            if variable in {"humidity_pct", "cloud_pct"} and not -1e-6 <= value <= 100 + 1e-6:
                dropped += 1
                continue
            valid = item["init_time"] + pd.Timedelta(hours=start if interval else end)
            valid_end = item["init_time"] + pd.Timedelta(hours=end)
            output.append({**_base("forecast", source, model, now, publication, confidence), "model_version": item.get("model_version", "unknown"), "variable": variable, "location_id": item["location_id"], "latitude": item["latitude"], "longitude": item["longitude"], "init_time": item["init_time"], "valid_time": valid, "valid_end": valid_end, "member": item["member"], "value": value, "unit": UNITS[variable]})
    # Vector magnitude is derived only from matched components at the same instant.
    winds = {}
    for row in output:
        if row["variable"] in {"wind_u_ms", "wind_v_ms"}:
            key = (row["location_id"], row["init_time"], row["valid_time"], row["valid_end"], row["member"])
            winds.setdefault(key, {})[row["variable"]] = row
    for components in winds.values():
        if {"wind_u_ms", "wind_v_ms"} <= components.keys():
            u, v = components["wind_u_ms"], components["wind_v_ms"]
            output.append({**u, "variable": "wind_speed_ms", "value": float(np.hypot(u["value"], v["value"]))})
    # ECMWF supplies dewpoint rather than relative humidity in the selected subset.
    # Magnus over-water approximation; preserve any native RH values when present.
    temperatures, humidity_keys = {}, set()
    for row in output:
        key = (row["location_id"], row["init_time"], row["valid_time"], row["valid_end"], row["member"])
        if row["variable"] in {"temperature_c", "dewpoint_c"}:
            temperatures.setdefault(key, {})[row["variable"]] = row
        elif row["variable"] == "humidity_pct":
            humidity_keys.add(key)
    for key, values in temperatures.items():
        if key not in humidity_keys and {"temperature_c", "dewpoint_c"} <= values.keys():
            temperature, dewpoint = values["temperature_c"], values["dewpoint_c"]
            t, d = temperature["value"], dewpoint["value"]
            if min(t, d) > -100:
                humidity = 100 * np.exp(17.625 * d / (243.04 + d) - 17.625 * t / (243.04 + t))
                if 0 <= humidity <= 100 + 1e-6:
                    output.append({**temperature, "variable": "humidity_pct", "unit": "%", "value": float(min(100, humidity))})
    return output, dropped


def parse_grib(path, source, model, init_time=None, available_at=None, availability_confidence="unknown") -> SourceBatch:
    """Import operational/archive GRIB, including explicitly downloaded NCAR files.

    No network is used. Crop to Singapore's regional box, preserve each native
    member and lead (0..72h), and retain GRIB accumulation intervals. Availability
    evidence must be supplied independently of file modification/init timestamps.
    """
    import cfgrib
    path = Path(path)
    payload = path.read_bytes()
    now = _time()
    if availability_confidence not in {"observed", "published", "reconstructed", "unknown"}:
        raise ValueError("Unknown availability confidence")
    if available_at is not None and availability_confidence == "unknown":
        availability_confidence = "published"
    if available_at is None and availability_confidence != "unknown":
        raise ValueError("Known availability confidence requires explicit available_at")
    publication = _time(available_at, explicit=True) if available_at is not None else pd.NaT
    expected_init = _time(init_time, explicit=True) if init_time is not None else None
    inventory = _grib_inventory(path)
    if not inventory:
        raise ValueError("File contains no GRIB messages")
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    raw_rows, unsupported, missing = [], set(), 0
    try:
        for dataset in datasets:
            if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
                continue
            south, north, west, east = BOUNDS
            dataset = dataset.assign_coords(longitude=((dataset.longitude + 180) % 360) - 180)
            dataset = dataset.where((dataset.latitude >= south) & (dataset.latitude <= north) & (dataset.longitude >= west) & (dataset.longitude <= east), drop=True)
            for name, array in dataset.data_vars.items():
                short = str(array.attrs.get("GRIB_shortName", name))
                variable = GRIB_FIELDS.get(short)
                if variable is None:
                    unsupported.add(short)
                    continue
                table = array.to_dataframe(name="value").reset_index()
                for item in table.to_dict("records"):
                    value = float(item["value"])
                    if not np.isfinite(value):
                        missing += 1
                        continue
                    initialization = _time(item["time"])
                    if expected_init is not None and initialization != expected_init:
                        raise ValueError("GRIB initialization differs from requested cycle")
                    lead = pd.Timedelta(item.get("step", pd.Timedelta(0))).total_seconds() / 3600
                    if not 0 <= lead <= 72:
                        continue
                    member = str(int(item.get("number", 0)))
                    details = inventory.get((short, initialization.value, lead, member))
                    if details is None:
                        raise ValueError("GRIB interval metadata could not be matched to a decoded field")
                    latitude, longitude = float(item["latitude"]), float(item["longitude"])
                    raw_rows.append({"variable": variable, "location_id": f"grid:{latitude:.5f}:{longitude:.5f}", "latitude": latitude, "longitude": longitude, "init_time": initialization, "member": member, "value": value, "start": details["start"], "end": details["end"], "step_type": details["step_type"], "raw_unit": details["unit"]})
    finally:
        for dataset in datasets:
            dataset.close()
    rows, excluded = _normalize_grib_rows(raw_rows, source=source, model=model, now=now, publication=publication, confidence=availability_confidence)
    if not rows:
        raise ValueError("No supported weather fields in Singapore bounds and leads 0..72")
    records = _records(rows)
    # Forecast files can include duplicate messages; keep identical facts only once.
    records = records.drop_duplicates(["variable", "location_id", "init_time", "valid_time", "valid_end", "member", "value"])
    return _batch(payload, path.name, {"source": source, "model": model, "bounds": list(BOUNDS), "lead_range_hours": [0, 72], "excluded_records": missing + excluded, "unsupported_short_names": sorted(unsupported), "native_intervals_preserved": True, "availability_confidence": availability_confidence, "available_at": publication.isoformat() if not pd.isna(publication) else None, "model_version": "unknown"}, records)
