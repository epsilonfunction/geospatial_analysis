"""Opt-in, local-only flexible-demand benchmark for permitted price data.

No prices are downloaded here. WEP measures wholesale costs, not the consumer's
complete bill. USEP results measure the energy component only. Schedules assume
zero switching cost and zero market impact. Publication and finality come from
the supplied records; this module never infers settlement from a calendar lag.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from typing import Iterable
from zoneinfo import ZoneInfo

SG = ZoneInfo("Asia/Singapore")
UTC = timezone.utc
ASSUMPTIONS = "4 MWh/day; maximum 1 MW; zero switching cost; zero market impact"


def _utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamps require an explicit timezone")
    return value.astimezone(UTC)


def _day(value) -> date:
    if isinstance(value, str):
        return date.fromisoformat(value)
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError("delivery_date must be a date, not a datetime")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _permission(permitted_price_data: bool):
    if permitted_price_data is not True:
        raise PermissionError("Price analytics require explicit permitted_price_data=True for data you may use")


def _basis(value: str) -> str:
    if value not in {"WEP", "USEP"}:
        raise ValueError("price_basis must be WEP or USEP")
    return value


def _rows(data) -> list[dict]:
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return [dict(row) for row in data]


def decision_deadline(delivery_date) -> datetime:
    """Noon Singapore time on the calendar day before delivery, in UTC."""
    return datetime.combine(_day(delivery_date) - timedelta(days=1), time(12), SG).astimezone(UTC)


def _decision(delivery_date: date, decision_time=None) -> datetime:
    deadline = decision_deadline(delivery_date)
    decision = deadline if decision_time is None else _utc(decision_time)
    if decision > deadline or decision.astimezone(SG).date() != delivery_date - timedelta(days=1):
        raise ValueError("Decision must occur on the previous Singapore day, no later than noon")
    return decision


def operating_intervals(delivery_date) -> tuple[datetime, ...]:
    start = datetime.combine(_day(delivery_date), time(8), SG).astimezone(UTC)
    return tuple(start + timedelta(minutes=30 * i) for i in range(24))


@dataclass(frozen=True)
class ConsumptionSlot:
    target_time: datetime
    power_mw: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "target_time", _utc(self.target_time))
        power = _finite(self.power_mw, "power_mw")
        if power != 1.0:
            raise ValueError("This benchmark uses exactly 1 MW in each selected interval")
        object.__setattr__(self, "power_mw", power)


@dataclass(frozen=True)
class ConsumptionSchedule:
    delivery_date: date
    decision_time: datetime
    slots: tuple[ConsumptionSlot, ...]
    strategy: str
    price_basis: str = "WEP"
    input_digest: str = "fixed-calendar-schedule"

    def __post_init__(self):
        delivery = _day(self.delivery_date)
        object.__setattr__(self, "delivery_date", delivery)
        object.__setattr__(self, "decision_time", _decision(delivery, self.decision_time))
        object.__setattr__(self, "slots", tuple(self.slots))
        _basis(self.price_basis)
        if len(self.slots) != 8 or any(not isinstance(slot, ConsumptionSlot) for slot in self.slots):
            raise ValueError("Schedule requires eight half-hour 1 MW slots (exactly 4 MWh)")
        targets = tuple(slot.target_time for slot in self.slots)
        if len(set(targets)) != 8 or targets != tuple(sorted(targets)):
            raise ValueError("Schedule slots must be unique and in chronological order")
        if not set(targets).issubset(operating_intervals(delivery)):
            raise ValueError("Every slot must lie within 08:00–20:00 Singapore time")

    @property
    def energy_mwh(self) -> float:
        return sum(slot.power_mw * 0.5 for slot in self.slots)

    def to_dict(self) -> dict:
        return {
            "delivery_date": self.delivery_date.isoformat(),
            "decision_time": self.decision_time.isoformat(),
            "slots": [{"target_time": s.target_time.isoformat(), "power_mw": s.power_mw} for s in self.slots],
            "strategy": self.strategy, "price_basis": self.price_basis,
            "input_digest": self.input_digest, "assumptions": ASSUMPTIONS,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def schedule_id(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, payload: str) -> "ConsumptionSchedule":
        data = json.loads(payload)
        data.pop("assumptions", None)
        data["slots"] = tuple(ConsumptionSlot(**slot) for slot in data["slots"])
        return cls(**data)


def fixed_schedule(delivery_date, decision_time=None, price_basis="WEP") -> ConsumptionSchedule:
    """Calendar baseline: consume 1 MW from 12:00 through 16:00 Singapore time."""
    delivery = _day(delivery_date)
    start = datetime.combine(delivery, time(12), SG).astimezone(UTC)
    return ConsumptionSchedule(delivery, _decision(delivery, decision_time), tuple(
        ConsumptionSlot(start + timedelta(minutes=30 * i)) for i in range(8)
    ), "fixed_12_16", price_basis)


def optimize_schedule(
    forecasts, delivery_date, *, permitted_price_data=False, decision_time=None,
    price_basis="WEP", strategy="weather",
) -> ConsumptionSchedule:
    """Freeze the eight cheapest intervals using an already frozen prediction set.

    Rows require target_time, mean, and available_at (or decision_time for saved
    predictions). Supply exactly one frozen forecast per operating interval;
    duplicates and predictions available after the decision are rejected.
    Full-day predictions are accepted; out-of-window rows are not selected.
    """
    _permission(permitted_price_data)
    _basis(price_basis)
    delivery = _day(delivery_date)
    decision = _decision(delivery, decision_time)
    operating = set(operating_intervals(delivery))
    prices = {}
    records = []
    seen = set()
    for row in _rows(forecasts):
        target = _utc(row["target_time"])
        if target in seen:
            raise ValueError("Duplicate forecast interval; provide a frozen prediction set")
        seen.add(target)
        local = target.astimezone(SG)
        if local.date() != delivery or local.minute not in {0, 30} or local.second or local.microsecond:
            raise ValueError("Forecast target_time must be a half-hour boundary on delivery_date")
        available = row.get("available_at", row.get("decision_time"))
        if available is None or _utc(available) > decision:
            raise ValueError("Forecast availability must be known and no later than the decision")
        for clock in ("decision_time", "published_at"):
            if clock in row and _utc(row[clock]) > decision:
                raise ValueError("Forecast decision and publication clocks must not follow the frozen decision")
        if row.get("price_basis", price_basis) != price_basis:
            raise ValueError("Mixed price bases are not supported")
        price = _finite(row["mean"], "mean")
        if target in operating:
            prices[target] = price
            records.append({"target_time": target.isoformat(), "mean": price, "available_at": _utc(available).isoformat()})
    if set(prices) != operating:
        raise ValueError("Forecasts must cover all 24 half-hour operating intervals")
    chosen = sorted(sorted(prices, key=lambda target: (prices[target], target))[:8])
    digest = sha256(json.dumps(sorted(records, key=lambda r: r["target_time"]), sort_keys=True).encode()).hexdigest()
    return ConsumptionSchedule(delivery, decision, tuple(ConsumptionSlot(t) for t in chosen), strategy, price_basis, digest)


def save_schedule(schedule: ConsumptionSchedule, path) -> Path:
    """Write once. Re-saving an identical schedule is safe; revisions cannot overwrite it."""
    destination = Path(path)
    payload = schedule.to_json()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if destination.read_text("utf-8") != payload:
            raise ValueError("A different frozen schedule already exists at this path") from None
    return destination


@dataclass(frozen=True)
class EconomicResult:
    delivery_date: date
    as_of: datetime
    status: str
    reason: str
    price_basis: str
    weather_cost_sgd: float | None = None
    fixed_cost_sgd: float | None = None
    no_weather_cost_sgd: float | None = None
    savings_vs_fixed_sgd: float | None = None
    savings_vs_no_weather_sgd: float | None = None
    settlement_digest: str | None = None
    assumptions: str = ASSUMPTIONS


def evaluate_schedules(
    weather: ConsumptionSchedule, no_weather: ConsumptionSchedule, settlements,
    *, as_of, permitted_price_data=False,
) -> EconomicResult:
    """Score only complete, final settlement records actually published by as_of.

    Rows require target_time, price_sgd_mwh, status, published_at. Missing rows,
    provisional rows, and unpublished final rows return pending without partial
    costs. Duplicate or malformed rows raise ValueError. Store successive results
    separately to retain the first and revised settlement scores.
    """
    _permission(permitted_price_data)
    cutoff = _utc(as_of)
    if (weather.delivery_date, weather.decision_time, weather.price_basis) != (
        no_weather.delivery_date, no_weather.decision_time, no_weather.price_basis
    ):
        raise ValueError("Comparison schedules must share delivery date, decision time, and price basis")
    fixed = fixed_schedule(weather.delivery_date, weather.decision_time, weather.price_basis)
    required = {slot.target_time for schedule in (weather, no_weather, fixed) for slot in schedule.slots}
    parsed = {}
    for row in _rows(settlements):
        target = _utc(row["target_time"])
        local = target.astimezone(SG)
        if local.date() != weather.delivery_date or local.minute not in {0, 30} or local.second or local.microsecond:
            raise ValueError("Settlement target_time must be a half-hour boundary on delivery_date")
        if target in parsed:
            raise ValueError("Duplicate settlement interval; supply one settlement vintage")
        if row.get("price_basis", weather.price_basis) != weather.price_basis:
            raise ValueError("Settlement price basis differs from schedule")
        price = _finite(row["price_sgd_mwh"], "price_sgd_mwh")
        status = row["status"]
        if status not in {"final", "provisional", "preliminary"}:
            raise ValueError("Settlement status must be final, provisional, or preliminary")
        published = _utc(row["published_at"])
        if status == "final" and published < target + timedelta(minutes=30):
            raise ValueError("Final settlement cannot precede delivery interval completion")
        parsed[target] = (price, status, published)
    common = dict(delivery_date=weather.delivery_date, as_of=cutoff, price_basis=weather.price_basis)
    missing = required - set(parsed)
    if missing:
        return EconomicResult(**common, status="pending", reason=f"Missing {len(missing)} required settlement intervals")
    unavailable = [t for t in required if parsed[t][1] != "final" or parsed[t][2] > cutoff]
    if unavailable:
        return EconomicResult(**common, status="pending", reason=f"{len(unavailable)} intervals lack final prices published by as_of")
    def cost(schedule):
        return math.fsum(slot.power_mw * 0.5 * parsed[slot.target_time][0] for slot in schedule.slots)
    weather_cost, fixed_cost, no_weather_cost = map(cost, (weather, fixed, no_weather))
    snapshot = [(target.isoformat(), parsed[target][0], parsed[target][2].isoformat()) for target in sorted(required)]
    return EconomicResult(
        **common, status="settled", reason="Wholesale cost benchmark" if weather.price_basis == "WEP" else "Energy-component savings only",
        weather_cost_sgd=weather_cost, fixed_cost_sgd=fixed_cost, no_weather_cost_sgd=no_weather_cost,
        savings_vs_fixed_sgd=fixed_cost - weather_cost, savings_vs_no_weather_sgd=no_weather_cost - weather_cost,
        settlement_digest=sha256(json.dumps(snapshot).encode()).hexdigest(),
    )


def summarize_economics(results: Iterable[EconomicResult], *, bootstrap_samples=1000, random_seed=0) -> dict:
    """Summarize one scoring vintage per delivery day, retaining pending counts."""
    rows = list(results)
    if len({r.delivery_date for r in rows}) != len(rows):
        raise ValueError("Select one scoring vintage per day before aggregating")
    if len({r.price_basis for r in rows}) > 1:
        raise ValueError("Do not mix WEP and USEP in an economic report")
    settled = [r for r in rows if r.status == "settled"]
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an integer of at least 100")
    summary = {"days": len(rows), "settled_days": len(settled), "pending_days": len(rows) - len(settled),
               "price_basis": rows[0].price_basis if rows else None, "assumptions": ASSUMPTIONS,
               "claim": "Illustrative benchmark; held-out economic improvement is not established by this summary"}
    for baseline in ("fixed", "no_weather"):
        savings = [getattr(row, f"savings_vs_{baseline}_sgd") for row in settled]
        summary[f"total_savings_vs_{baseline}_sgd"] = math.fsum(savings) if savings else None
        summary[f"mean_daily_savings_vs_{baseline}_sgd"] = math.fsum(savings) / len(savings) if savings else None
        summary[f"worst_day_additional_cost_vs_{baseline}_sgd"] = max(0.0, -min(savings)) if savings else None
        weeks = {}
        for row in sorted(settled, key=lambda item: item.delivery_date):
            iso = row.delivery_date.isocalendar()
            weeks.setdefault((iso.year, iso.week), []).append(getattr(row, f"savings_vs_{baseline}_sgd"))
        blocks = list(weeks.values())
        summary["observed_week_blocks"] = len(blocks)
        interval = None
        if len(blocks) >= 2:
            random = Random(random_seed)
            estimates = []
            for _ in range(bootstrap_samples):
                sample = [value for _ in blocks for value in random.choice(blocks)]
                estimates.append(math.fsum(sample) / len(sample))
            estimates.sort()
            interval = [estimates[int((bootstrap_samples - 1) * .025)], estimates[int((bootstrap_samples - 1) * .975)]]
        summary[f"mean_daily_savings_vs_{baseline}_95pct_week_block_interval_sgd"] = interval
    summary["uncertainty_method"] = "Seeded resampling of observed ISO-week blocks; missing days are not imputed; fewer than two weeks gives no interval"
    return summary
