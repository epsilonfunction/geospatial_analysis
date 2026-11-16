"""Read-only local dashboard.

Launch: streamlit run src/weather_engine/dashboard.py -- --data-dir data/weather_engine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _signature(paths):
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def main():
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/weather_engine")
    args, _ = parser.parse_known_args()
    root = Path(args.data_dir).expanduser().resolve()
    st.set_page_config(page_title="Singapore weather analytics", layout="wide")
    st.title("Singapore weather analytics")
    st.caption("Forecast evidence and operational status · times displayed in Asia/Singapore")
    st.sidebar.caption(f"Read-only archive: {root}")
    st.sidebar.button("Refresh", on_click=st.cache_data.clear)
    st.sidebar.info("Weather and demand validation. Economic performance requires separately permitted price data and held-out evaluation.")

    @st.cache_data(show_spinner=False)
    def read_frames(signature):
        frames, errors = [], []
        for name, _, _ in signature:
            try:
                frames.append(pd.read_parquet(name))
            except (OSError, ValueError, ImportError) as exc:
                errors.append(f"{Path(name).name}: {exc}")
        return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), errors

    def load(paths):
        frame, errors = read_frames(_signature(paths))
        for error in errors:
            st.warning(error)
        return frame

    def local_times(frame):
        result = frame.copy()
        for col in result:
            if col.endswith("_time") or col.endswith("_at") or col in {"valid_end", "target_end"}:
                try:
                    result[col] = pd.to_datetime(result[col], utc=True, errors="raise").dt.tz_convert("Asia/Singapore")
                except (ValueError, TypeError):
                    pass
        return result

    def table(frame):
        st.dataframe(local_times(frame), hide_index=True, use_container_width=True)

    if not root.exists():
        st.info("No archive exists yet. Ingest forecast and observation data, then run a backtest or forward prediction.")
        st.code("weather-engine --help\nweather-engine ingest --help\nweather-engine backtest --help\nweather-engine forward --help")
        return

    record_files = sorted((root / "records").glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    limit = st.sidebar.number_input("Recent record fragments to display", min_value=1, max_value=5000, value=200)
    records = load(record_files[:limit])
    manifests = sorted((root / "runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not record_files and not manifests:
        st.info("The archive has no records or prediction runs yet.")
        st.code("weather-engine ingest --help\nweather-engine forward --help")
        return
    if len(record_files) > limit:
        st.caption(f"Coverage and freshness below use the newest {limit} of {len(record_files)} fragments; this is a display sample, not a full archive audit.")

    freshness_tab, predictions_tab, scores_tab, audit_tab = st.tabs(["Sources and forecasts", "Predictions", "Verification", "Run evidence"])
    with freshness_tab:
        if records.empty:
            st.info("No normalized weather or demand records are available.")
        else:
            group_cols = [c for c in ["source", "model", "kind"] if c in records]
            clock = next((c for c in ["first_seen_at", "ingested_at", "available_at"] if c in records), None)
            st.subheader("Collection freshness")
            if group_cols and clock:
                fresh = records.copy()
                fresh[clock] = pd.to_datetime(fresh[clock], utc=True, errors="coerce")
                fresh = fresh.groupby(group_cols, dropna=False)[clock].max().reset_index()
                fresh["hours_since_collection"] = (pd.Timestamp.now(tz="UTC") - fresh[clock]).dt.total_seconds() / 3600
                table(fresh)
            else:
                st.info("Collection timestamps are unavailable in these records.")
            st.subheader("Available coverage")
            coverage_cols = [c for c in ["source", "model", "variable", "location_id"] if c in records]
            if coverage_cols and "valid_time" in records:
                coverage = records.groupby(coverage_cols, dropna=False).agg(
                    rows=("valid_time", "size"), first_valid_time=("valid_time", "min"), last_valid_time=("valid_time", "max"),
                ).reset_index()
                table(coverage)
                st.caption("Presence of rows is not proof of complete expected cycles, fields, or members. Use the saved source audit for missingness.")
            if "availability_confidence" in records:
                table(records["availability_confidence"].value_counts(dropna=False).rename_axis("availability_confidence").reset_index(name="records"))
            if {"kind", "variable", "model", "valid_time", "value"}.issubset(records.columns):
                forecasts = records.loc[records.kind.eq("forecast")].copy()
                if not forecasts.empty:
                    st.subheader("Latest forecast cycles and model disagreement")
                    variable = st.selectbox("Weather variable", sorted(forecasts.variable.dropna().unique()))
                    forecasts = forecasts.loc[forecasts.variable.eq(variable)]
                    if "location_id" in forecasts:
                        location = st.selectbox("Forecast location", sorted(forecasts.location_id.dropna().unique()))
                        forecasts = forecasts.loc[forecasts.location_id.eq(location)]
                    if "init_time" in forecasts:
                        forecasts["init_time"] = pd.to_datetime(forecasts.init_time, utc=True, errors="coerce")
                        forecasts = forecasts.loc[forecasts.init_time.eq(forecasts.groupby("model").init_time.transform("max"))]
                    compare = forecasts.groupby(["valid_time", "model"], dropna=False).value.mean().unstack("model").sort_index()
                    compare.index = pd.to_datetime(compare.index, utc=True).tz_convert("Asia/Singapore")
                    st.line_chart(compare)
                    common = compare.dropna()
                    if compare.shape[1] > 1 and not common.empty:
                        st.metric("Mean model spread at common valid times", f"{(common.max(axis=1) - common.min(axis=1)).mean():.3f}")
                    st.caption("Ensemble members are averaged within each model. Only shared valid times contribute to model spread; native time intervals are retained.")

    selected_manifest = None
    predictions = pd.DataFrame()
    if manifests:
        chosen = st.sidebar.selectbox("Prediction run", manifests, format_func=lambda p: p.parent.name)
        try:
            selected_manifest = json.loads(chosen.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            st.warning(f"Cannot read run manifest: {exc}")
        prediction_path = chosen.parent / "predictions.parquet"
        if prediction_path.exists():
            predictions = load([prediction_path])
    with predictions_tab:
        if predictions.empty:
            st.info("Select a completed backtest or forward run to inspect its predictions.")
        else:
            if "status" in predictions:
                status = predictions.status.value_counts()
                a, b, c = st.columns(3)
                a.metric("Prediction rows", len(predictions))
                b.metric("Fallback rows", f"{status.get('fallback', 0) / len(predictions):.1%}")
                c.metric("Unavailable rows", f"{status.get('unavailable', 0) / len(predictions):.1%}")
                st.caption("Row rates are shown separately from the daily deadline service level; they do not establish the 99% timely-day target.")
            view = predictions.copy()
            for col, label in [("variable", "Prediction target"), ("location_id", "Prediction location"), ("model", "Prediction model")]:
                if col in view:
                    options = sorted(view[col].dropna().astype(str).unique())
                    if options:
                        choice = st.selectbox(label, options)
                        view = view.loc[view[col].astype(str).eq(choice)]
            if {"target_time", "mean"}.issubset(view.columns):
                view = local_times(view).sort_values("target_time")
                figure = go.Figure()
                if {"p10", "p90"}.issubset(view.columns):
                    figure.add_trace(go.Scatter(x=view.target_time, y=view.p90, mode="lines", line={"width": 0}, name="P90"))
                    figure.add_trace(go.Scatter(x=view.target_time, y=view.p10, mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(45,140,220,.18)", name="P10–P90"))
                figure.add_trace(go.Scatter(x=view.target_time, y=view["mean"], mode="lines", name="Mean"))
                if "p50" in view:
                    figure.add_trace(go.Scatter(x=view.target_time, y=view.p50, mode="lines", line={"dash": "dot"}, name="P50"))
                figure.update_layout(xaxis_title="Delivery time (Singapore)", yaxis_title="Prediction", legend={"orientation": "h"})
                st.plotly_chart(figure, use_container_width=True)
            if "reason" in predictions:
                reasons = predictions.loc[predictions.reason.fillna("").ne(""), [c for c in ["decision_time", "model", "status", "reason"] if c in predictions]].drop_duplicates()
                if not reasons.empty:
                    st.subheader("Missing inputs and fallback reasons")
                    table(reasons)
            with st.expander("Prediction records"):
                table(view)

    with scores_tab:
        score_paths = sorted(root.glob("scores*.parquet"))
        revisions = []
        if manifests:
            revisions = sorted((chosen.parent / "verification").glob("*/scores.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
            if revisions:
                revision = st.selectbox("Verification revision", revisions, format_func=lambda p: p.parent.name)
                score_paths = [revision]
                st.caption("One saved scoring revision is shown at a time so later labels do not overwrite or double-count earlier scores.")
            else:
                score_paths += sorted(chosen.parent.glob("scores*.parquet"))
        if not revisions and (root / "scores").exists():
            score_paths += sorted((root / "scores").glob("*.parquet"))
        scores = load(list(dict.fromkeys(score_paths))) if score_paths else pd.DataFrame()
        if scores.empty:
            st.info("No verification scores are available yet. Forecasts remain unverified until observations or published demand labels arrive.")
        else:
            st.caption("Scores are displayed with their saved metadata. Historical reconstructed replay must be assessed separately from strict availability replay.")
            table(scores)
            if {"model", "metric", "value"}.issubset(scores.columns):
                metric = st.selectbox("Verification metric", sorted(scores.metric.dropna().unique()))
                subset = scores.loc[scores.metric.eq(metric)]
                st.bar_chart(subset.groupby("model").value.mean())
                st.caption("Mean of saved metric rows; consult record weights and evaluation slices before comparing models.")

    with audit_tab:
        if selected_manifest is not None:
            config = selected_manifest.get("config") or {}
            mode = selected_manifest.get("mode", selected_manifest.get("replay_mode", config.get("mode", "not recorded")))
            st.write(f"Availability mode: **{mode}**")
            if "reconstructed" in str(selected_manifest).lower():
                st.warning("This run includes reconstructed availability. It is not evidence of verified historical operational performance.")
            st.json(selected_manifest)
        audit_paths = sorted((root / "audits").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in audit_paths[:5]:
            with st.expander(f"Source audit: {path.name}"):
                try:
                    st.json(json.loads(path.read_text("utf-8")))
                except (OSError, ValueError) as exc:
                    st.warning(str(exc))
        st.caption("The dashboard reads archive artifacts only. Refresh does not collect sources, train models, change frozen inputs, or revise predictions.")


if __name__ == "__main__":
    main()
