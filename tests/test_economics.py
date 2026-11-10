from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import json

import pytest

from weather_engine.economics import (
    ConsumptionSchedule, ConsumptionSlot, decision_deadline, evaluate_schedules,
    fixed_schedule, operating_intervals, optimize_schedule, save_schedule, summarize_economics,
)

DAY = date(2024, 1, 2)
DECISION = decision_deadline(DAY)


def forecast(prices=None):
    return [dict(target_time=t, mean=(prices or list(range(24)))[i], available_at=DECISION)
            for i, t in enumerate(operating_intervals(DAY))]


def schedule(prices=None, **kwargs):
    return optimize_schedule(forecast(prices), DAY, permitted_price_data=True, **kwargs)


def settlements():
    return [dict(target_time=t, price_sgd_mwh=i * 10, status="final", published_at=DECISION + timedelta(days=10))
            for i, t in enumerate(operating_intervals(DAY))]


def evaluate(rows=None, **kwargs):
    weather = schedule()
    no_weather = schedule(list(reversed(range(24))), strategy="no_weather")
    return evaluate_schedules(weather, no_weather, settlements() if rows is None else rows,
                              as_of=kwargs.pop("as_of", DECISION + timedelta(days=11)),
                              permitted_price_data=True, **kwargs)


def test_requires_explicit_permission_before_touching_prices():
    with pytest.raises(PermissionError):
        optimize_schedule(None, DAY)
    with pytest.raises(PermissionError):
        evaluate_schedules(None, None, None, as_of=DECISION)


def test_physical_constraints_and_chronological_ties():
    selected = schedule([4] * 24)
    assert selected.energy_mwh == 4
    assert len(selected.slots) == 8
    assert tuple(s.target_time for s in selected.slots) == operating_intervals(DAY)[:8]
    assert all(s.power_mw == 1 for s in selected.slots)
    assert selected.decision_time.isoformat() == "2024-01-01T04:00:00+00:00"
    assert selected.slots[0].target_time.isoformat() == "2024-01-02T00:00:00+00:00"


def test_noncontiguous_and_negative_prices():
    selected = schedule([-10 if i % 3 == 0 else 100 for i in range(24)])
    assert [s.target_time for s in selected.slots] == list(operating_intervals(DAY)[::3])


def test_fixed_baseline_and_nested_immutability(tmp_path):
    fixed = fixed_schedule(DAY)
    assert fixed.slots[0].target_time.hour == 4
    assert fixed.slots[-1].target_time.hour == 7
    assert fixed.slots[-1].target_time.minute == 30
    with pytest.raises(FrozenInstanceError):
        fixed.strategy = "changed"
    with pytest.raises(FrozenInstanceError):
        fixed.slots[0].power_mw = 0
    assert ConsumptionSchedule.from_json(fixed.to_json()) == fixed
    assert ConsumptionSchedule.from_json(fixed.to_json()).schedule_id == fixed.schedule_id
    path = save_schedule(fixed, tmp_path / "schedule.json")
    assert save_schedule(fixed, path) == path
    with pytest.raises(ValueError, match="different frozen"):
        save_schedule(schedule(), path)
    assert json.loads(path.read_text())["strategy"] == "fixed_12_16"


@pytest.mark.parametrize("changes", [
    {"mean": float("nan")}, {"mean": float("inf")}, {"mean": True},
    {"available_at": DECISION + timedelta(microseconds=1)}, {"available_at": None},
    {"target_time": DECISION.replace(tzinfo=None)},
    {"target_time": operating_intervals(DAY)[0] + timedelta(minutes=1)},
    {"price_basis": "USEP"},
])
def test_invalid_forecasts(changes):
    rows = forecast()
    rows[0].update(changes)
    with pytest.raises(ValueError):
        optimize_schedule(rows, DAY, permitted_price_data=True)


def test_incomplete_and_duplicate_forecasts():
    rows = forecast()
    for invalid in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError):
            optimize_schedule(invalid, DAY, permitted_price_data=True)
    with pytest.raises(ValueError, match="no later than noon"):
        schedule(decision_time=DECISION + timedelta(seconds=1))


def test_schedule_cannot_violate_constraints():
    fixed = fixed_schedule(DAY)
    with pytest.raises(ValueError):
        replace(fixed, slots=fixed.slots[:7])
    with pytest.raises(ValueError):
        replace(fixed, slots=(fixed.slots[0],) * 8)
    with pytest.raises(ValueError):
        replace(fixed, slots=(ConsumptionSlot(operating_intervals(DAY)[0] - timedelta(minutes=30)),) + fixed.slots[1:])
    with pytest.raises(ValueError):
        ConsumptionSlot(operating_intervals(DAY)[0], 1.1)


def test_correct_settlement_cost_and_two_baselines():
    result = evaluate()
    assert result.status == "settled"
    assert result.weather_cost_sgd == sum(range(8)) * 10 * .5
    assert result.fixed_cost_sgd == sum(range(8, 16)) * 10 * .5
    assert result.no_weather_cost_sgd == sum(range(16, 24)) * 10 * .5
    assert result.savings_vs_fixed_sgd == 320
    assert result.savings_vs_no_weather_sgd == 640
    assert result.settlement_digest


def test_settlement_is_pending_until_every_selected_interval_is_final_and_published():
    rows = settlements()
    assert evaluate(rows[:-1]).status == "pending"
    assert evaluate(rows, as_of=DECISION + timedelta(days=9)).status == "pending"
    rows[0]["status"] = "provisional"
    pending = evaluate(rows)
    assert pending.status == "pending"
    assert pending.weather_cost_sgd is None
    assert pending.savings_vs_fixed_sgd is None


@pytest.mark.parametrize("changes", [
    {"price_sgd_mwh": float("nan")}, {"price_sgd_mwh": "bad"}, {"status": "unknown"},
    {"published_at": DECISION}, {"published_at": DECISION.replace(tzinfo=None)}, {"price_basis": "USEP"},
])
def test_malformed_settlement_rejected(changes):
    rows = settlements()
    rows[0].update(changes)
    with pytest.raises(ValueError):
        evaluate(rows)


def test_duplicate_settlement_rejected():
    rows = settlements()
    with pytest.raises(ValueError, match="Duplicate settlement"):
        evaluate(rows + rows[:1])


def test_revised_settlements_preserve_distinct_result_and_schedule():
    original = evaluate()
    rows = settlements()
    rows[0]["price_sgd_mwh"] = 100
    rows[0]["published_at"] += timedelta(days=1)
    revised = evaluate(rows)
    assert original.settlement_digest != revised.settlement_digest
    assert original.weather_cost_sgd == 140
    assert revised.weather_cost_sgd == 190


def test_summary_separates_pending_and_refuses_double_counted_vintages():
    settled = evaluate()
    pending = replace(evaluate([], as_of=DECISION), delivery_date=DAY + timedelta(days=1))
    report = summarize_economics([settled, pending])
    assert report["settled_days"] == 1
    assert report["pending_days"] == 1
    assert report["worst_day_additional_cost_vs_fixed_sgd"] == 0
    assert report["total_savings_vs_fixed_sgd"] == 320
    with pytest.raises(ValueError, match="one scoring vintage"):
        summarize_economics([settled, settled])


def test_usep_results_explicitly_labeled():
    result = evaluate_schedules(schedule(price_basis="USEP"), schedule(price_basis="USEP", strategy="no_weather"),
                                settlements(), as_of=DECISION + timedelta(days=11), permitted_price_data=True)
    assert result.reason == "Energy-component savings only"


def test_report_week_block_uncertainty_requires_multiple_weeks():
    first = evaluate()
    assert summarize_economics([first])["mean_daily_savings_vs_fixed_95pct_week_block_interval_sgd"] is None
    second = replace(first, delivery_date=DAY + timedelta(days=7), savings_vs_fixed_sgd=-20)
    report = summarize_economics([first, second])
    assert report["observed_week_blocks"] == 2
    assert report["worst_day_additional_cost_vs_fixed_sgd"] == 20
    assert report["mean_daily_savings_vs_fixed_95pct_week_block_interval_sgd"] == [-20, 320]
    assert report == summarize_economics([first, second])


def test_future_prediction_clock_cannot_hide_behind_earlier_availability():
    rows = forecast()
    rows[0]["decision_time"] = DECISION + timedelta(minutes=1)
    with pytest.raises(ValueError):
        optimize_schedule(rows, DAY, permitted_price_data=True)
