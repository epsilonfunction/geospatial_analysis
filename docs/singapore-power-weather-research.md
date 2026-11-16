# Weather forecasting for Singapore power-market research

Research date: 6 November 2026. Scope: Singapore electricity only. This is a research assessment and proposed direction, not an implemented or validated trading system. The companion [implementation plan](./singapore-power-weather-plan.md) specifies the architecture, resource limits, milestones, and acceptance criteria.

## Recommendation

Build a small, probabilistic weather-to-power research system. Consume forecasts produced by weather centres, calibrate them against Singapore observations, and test whether those improvements help forecast electricity demand and solar variability. Add price-risk modelling only when suitable market data and its publication history are available.

The initial product should answer: **What has changed in tomorrow's weather outlook, how uncertain is it, and what might that mean for Singapore's half-hourly power demand?** A later product can answer whether that information improves an executable procurement or trading decision.

Assumptions for planning: one researcher, this Windows laptop, public data initially, no confirmed EMC subscription or physical portfolio. No external hardware or cloud budget has been confirmed. These assumptions define an executable starting scope and can be changed without redesigning the core archive.

## 1. The forecasting paradigm

Weather forecasting is a chain: observations establish the present state; a forecast model projects possible future states; local calibration corrects systematic errors; an application model translates weather into demand, generation, and decisions. Skill at one step does not guarantee economic value at the last step.

| Approach | What it contributes | Singapore relevance | Project treatment |
|---|---|---|---|
| Climatology and persistence | Historical typical conditions, or continuation of current conditions | Strong low-cost checks against unnecessary complexity; persistence is useful at short lead times | Mandatory baselines |
| Global numerical weather prediction, or NWP | Physical atmosphere simulation initialised through data assimilation | Captures monsoon flow and broad temperature, moisture, cloud and rainfall evolution | Use published IFS and GFS forecasts |
| Regional convection-permitting NWP | Finer treatment of local convection and coastlines | Potentially valuable for storms and solar variability; needs boundary conditions and observation assimilation | Evaluate accessible outputs later; do not develop a local operational NWP service |
| Radar/satellite nowcasting | Extrapolates observed cloud or rain motion, sometimes blended with NWP | Relevant to the next few half-hours, especially solar ramps; storm initiation remains difficult | Conditional extension after machine-readable data access is established |
| AI global forecasting | Learns atmospheric evolution and can produce deterministic or ensemble forecasts | An additional upstream forecast source; global skill rankings do not establish local irradiance or trading skill | Benchmark published AIFS outputs later |
| Statistical post-processing | Corrects location, lead-time and regime-dependent forecast errors | Affordable way to exploit Singapore station observations | Main local modelling work |
| Probabilistic and decision-focused forecasting | Represents uncertainty, then evaluates decisions under possible outcomes | Matters where demand and solar errors coincide with tight supply | Build calibrated intervals first; coherent scenarios later |

MSS describes its forecast process as combining station, satellite, radar and other observations with numerical guidance and forecaster judgement. Local thunderstorms remain challenging. CCRS describes SINGV as a 1.5 km regional system with assimilation and ECMWF boundary conditions. These capabilities explain why simply downloading a finer global grid does not reproduce a local weather service. [MSS forecasting](https://www.weather.gov.sg/learn_forecast/) · [CCRS SINGV](https://ccrs.weather.gov.sg/singv-%E2%94%80-a-tropical-convective-scale-nwpnowcasting-capability-for-singapore/)

The GenCast research demonstrates the value of probabilistic machine-learning weather forecasts at global medium-range scales. Its published evaluation is evidence for a method, not proof of superiority for Singapore half-hourly load, cloud ramps, or the current versions of competing models. Training a global model, running a pretrained global model, and consuming its forecast output are three different resource commitments. This project should start with the third. [Price et al., Nature, 2025](https://pubmed.ncbi.nlm.nih.gov/39633054/)

### Choose methods by decision horizon

| Horizon | Weather approach | Singapore power application | Priority |
|---|---|---|---|
| 0–2 hours | Observations, persistence, later cloud/rain nowcasting | Solar ramps and short-term portfolio adjustments | Later, after reliable live collection |
| 2–6 hours | Observation-adjusted NWP; nowcast/NWP blend if available | Intraday demand revisions and operating exposure | Second release |
| 6–48 hours | Calibrated NWP, calendar effects, forecast uncertainty | Next-day demand and solar-risk briefing | Initial focus |
| 2–7 days | Ensemble scenarios and forecast revisions | Procurement planning and operational preparation | Extension after the first 72 hours work |
| Weeks–seasons | Climate anomalies and probabilistic regime outlooks | Longer-term exposure scenarios | Outside initial scope |

Singapore's heat, humidity, convection, monsoon transitions and Sumatra squalls matter. Evaluate errors by local hour and weather regime; an annual temperature average can hide poor performance during afternoon storms. Surrounding atmospheric context is useful, but outputs and market analysis remain Singapore-only. [MSS climate](https://www.weather.gov.sg/climate-climate-of-singapore) · [MSS weather systems](https://www.weather.gov.sg/learn_weather_systems/)

## 2. Where weather could create power-market value

USEP is Singapore's half-hourly wholesale energy price and responds to supply and demand. Singapore remains heavily reliant on imported natural gas. Consequently, a weather signal must be evaluated alongside supply availability and fuel conditions before being interpreted as a price signal. [EMA USEP explanation](https://www.ema.gov.sg/resources/faqs/electricity/electricity-prices/what-is-uniform-singapore-energy-price-usep-and-how-is-it-determined-does-ema-regulate-the-usep) · [Government natural-gas overview](https://www.meti.gov.sg/energy-and-carbon/energy-supply/natural-gas/)

The following mechanisms are hypotheses to estimate, not established coefficients or trading rules:

| Application | Mechanism and useful output | Main prerequisite | Assessment |
|---|---|---|---|
| Weather-sensitive system demand | Temperature, humidity, sunshine and lagged heat affect cooling demand; forecast half-hourly demand and uncertainty | Historical demand with documented semantics and sufficiently detailed weather history | Best first power application |
| Solar variability | Clouds and irradiance change output; quantify daylight uncertainty and ramps | Irradiance or metered PV truth, capacity history and location weights | High relevance; data is the main dependency |
| Demand forecast revisions | New weather runs may imply deviations from an earlier demand outlook | Forecast vintages from the relevant information cutoff | Strong candidate for incremental value |
| Conditional price-spike risk | A demand/solar surprise may matter more when available supply is tight | Prices, outages/availability and forecasts known before the decision | Later-stage research |
| Retail procurement or flexible consumption | Better forecasts can reduce uncertain energy costs under real contractual constraints | Actual exposure, contract basis and feasible flexibility | Suitable first economic experiment |
| Battery dispatch | Forecast distribution informs charging and discharge decisions | Asset limits, efficiency, degradation, relevant settlement prices and market access | Asset-specific extension |
| Generator offering or thermal capability | Weather may affect output capability and price exposure | Plant-specific data, availability and nodal settlement | Outside the public-data pilot |

A storm can simultaneously reduce cooling demand and suppress solar generation. The effects on residual demand oppose one another, and their size varies by time and location. Avoid a fixed rule such as “rain means lower power prices.”

For a consistently defined physical boundary, residual demand can be expressed as gross demand minus the generation being treated as non-dispatchable. **Do not subtract solar from a published demand series until its treatment of embedded generation is established.** EMC's demand-forecast definition explicitly excludes some components, including generation from exempted embedded generators. The EMA historical series and the EMC forecast must be reconciled before computing forecast errors or residual demand. [EMC Guide to Prices](https://www.nems.emcsg.com/Guide-To-Prices)

Keep settlement targets distinct: USEP measures the hub energy component; WEP incorporates additional wholesale components; generator revenues may depend on nodal prices. None automatically represents an end-consumer's complete bill. Store the price basis explicitly. This distinction also determines which cost model is appropriate. [EMC price definitions](https://www.nems.emcsg.com/Guide-To-Prices)

An initial useful deliverable is a forecast revision in MW with a calibrated interval and a clear information cutoff. Turning it into a position requires an actual tradable exposure, available quotes or offers, costs, and decision constraints. No futures liquidity, market access, or European-style day-ahead/intraday spread is assumed here.

## 3. Data feasibility and limitations

These findings are based on the official documentation inspected on the research date. Documentation access does not establish complete downloadable history, contractual permission, or reliable endpoint operation; those are first-stage checks.

| Data | Evidence and intended use | Limitation / action |
|---|---|---|
| NEA station observations | [data.gov.sg collection](https://data.gov.sg/collections/1459/view) supplies temperature, humidity, rainfall and wind; station-level readings can be as frequent as one minute | Audit actual cadence, missing stations, revisions and archive retrieval; station coverage dates do not prove uninterrupted history |
| MSS forecast products | [24-hour forecast API documentation](https://developers.data.gov.sg/datasets/d_ce2eb1e307bda31993c533285834ef2b/view) provides a local operational reference | Broad forecast categories and ranges are not measured half-hourly weather; save issue times |
| ECMWF IFS/AIFS | [Open-data documentation](https://www.ecmwf.int/en/forecasts/datasets/open-data) provides operational forecasts with attribution terms | The primary rolling archive retains only recent runs; start preserving selected forecasts immediately once collection is implemented |
| NOAA GFS | [NOMADS](https://nomads.ncep.noaa.gov/) and its [GRIB filter](https://nomads.ncep.noaa.gov/info.php?page=gribfilter) support regional forecast subsets | Verify fields, lead steps and publication latency; regional subsetting controls bandwidth, not forecast skill |
| Open-Meteo | [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) exposes individual initialisations and documents model-dependent history | A lightweight alternative, subject to usage terms and service entitlement; preserve model identity and interpolation metadata |
| EMA system demand | [Half-hourly system demand](https://www.ema.gov.sg/resources/statistics/half-hourly-system-demand-data) is a public historical target in MW | Inspect downloadable files, interval convention, embedded generation and release delay; do not assume a live demand feed |
| EMC price/demand forecasts | [Public market pages](https://www.nems.emcsg.com/en/nems-prices) and [subscription catalogue](https://www.home.emcsg.com/services/nems-services/data-subscription) describe access | Public display is not proof of a supported bulk API or permission to archive all data |
| Solar observations | [SERIS monitoring network](https://www.seris.nus.edu.sg/services/real-time-monitoring-system-of-irradiance/) describes 25 irradiance stations | The public map does not establish access to a downloadable historical numerical series; obtain usable truth before claiming solar accuracy |
| Solar capacity/profile | [EMA energy statistics](https://www.ema.gov.sg/resources/singapore-energy-statistics/chapter2) and [Solar Generation Profile](https://www.ema.gov.sg/consumer-information/solar/solar-generation-profile) provide context | The settlement profile uses historical irradiance; it is not actual live generation and cannot validate individual cloud-ramp forecasts |

### Details that materially change the design

**ECMWF resolution and timing.** The current documentation lists IFS open data at 0.25° and, from 20 October 2026, 0.1°. The finer feed is derived from the same forecast and is released about two hours later; AIFS remains at 0.25°. At Singapore's latitude, these grid spacings are roughly 28 km and 11 km, respectively. Start with the earlier feed and compare accuracy at a common information cutoff before accepting the extra latency and volume. The documented IFS and AIFS lead-step structures also differ. [ECMWF product documentation](https://confluence.ecmwf.int/spaces/DAC/pages/272310539/ECMWF+open+data+real-time+forecasts+from+IFS+and+AIFS)

**ECMWF history.** The direct open portal describes a rolling archive of 12 runs, approximately two to three days. Downloading today's data later cannot reconstruct exactly what a model user had available months ago. Longer archives require a separately verified source. [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data)

**EMC forecast retention.** Its subscription page lists 16 months of real-time data, one week of forecast data, and a delayed offer stack published at D+28. The historical power-market product excludes forecast data. Therefore, historical prices alone cannot reconstruct a forecast-aware trading experiment, and a delayed offer stack is not a contemporaneously available feature. As a budgeting reference, Package A currently lists S$1,035.50 per month on the 12-month option; confirm the quotation, entitlement and applicable charges before purchase. No subscription is required to begin the weather pilot. [EMC subscription catalogue](https://www.home.emcsg.com/services/nems-services/data-subscription)

**Historical weather products differ.** Reanalysis estimates realised past conditions and is useful for climatology or exploratory sensitivity studies. It is not a forecast available before delivery. A seamless forecast archive, a fixed-lead archive and a specific model-run archive also answer different questions. Prefer identifiable runs and explicit availability evidence for realistic evaluation. Open-Meteo documents IFS 9 km single runs from March 2024 and most other models from April 2026, but actual model/variable coverage must be checked. [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) · [Single Runs API](https://open-meteo.com/en/docs/single-runs-api)

**Service terms matter separately from upstream data licences.** Open-Meteo's free service is for non-commercial use; commercial use requires the appropriate subscription, and historical/ensemble access has plan-specific entitlements. Business trading research should not automatically be classified as non-commercial. Direct public-source adapters are the default where service eligibility is unresolved. [Open-Meteo pricing and terms of access](https://open-meteo.com/en/pricing)

## 4. Modelling strategy

Use a baseline ladder so that every increase in complexity has an observable benefit:

1. **Weather:** persistence and hour-of-day climatology; raw individual NWP sources; rolling lead/hour bias correction; a regularised blend; quantile models only after sufficient forecast-error history exists.
2. **Demand:** calendar-only regression; seasonal lag baseline when those lags were actually available; calendar plus raw forecast weather; calendar plus calibrated forecast weather. The last comparison isolates the value of weather calibration.
3. **Solar:** clear-sky or historical daylight baseline; irradiance forecast; a PV conversion model with capacity/tilt assumptions; measured-output calibration where metering exists. Until then, label output as irradiance or an illustrative PV scenario.
4. **Prices:** calendar and permissible recent-price baseline; supply-informed baseline; add weather-derived demand/solar variables. Start with calibrated spike probabilities or quantiles rather than a single precise price claim.

Suitable laptop models are regularised regression, spline-based regression, small gradient-boosted trees and quantile regression. Use a few explainable features before neural networks. Candidate features include temperature, dewpoint/humidity, recent heat, cloud/shortwave, rain probability, local half-hour, weekday, public holiday, and season. Estimate interactions and lag effects from data; do not import a universal MW-per-degree coefficient.

Compute any apparent-temperature or wet-bulb feature with a documented formula and applicability range. Wet-bulb temperature is not WBGT. Treat it as a candidate predictor, not an automatic measure of electricity demand.

Preserve native temporal semantics. Three-hour forecast rainfall totals must be de-accumulated and aligned correctly; accumulated radiation in J/m² needs conversion using the correct duration. Generating 30-minute rows from three-hour forecasts does not create 30-minute meteorological information. Track interpolation and evaluate weather at its supported resolution, then separately assess half-hourly demand predictions.

Two deterministic models are useful for disagreement analysis but do not form a well-calibrated probabilistic ensemble by themselves. Initially use empirical residual intervals conditioned on lead/hour, with minimum sample requirements. Later, propagate coherent ensemble members through both load and solar models and add model residual uncertainty. Preserve temporal and cross-variable dependence: independently sampling every half-hour can seriously distort ramps and daily energy risk.

## 5. What would count as evidence

All forecasts must be evaluated against the information available at their decision cutoff. Retain initialization time, forecast valid interval, source publication time if known, first local observation, ingestion time, and model/configuration version. Unknown historical publication times support a labelled reconstruction, not a claim that the forecast could have been traded.

Use chronological training, calibration and test windows. Fit preprocessing and model selection inside each training window. Exclude overlapping forecast targets around split boundaries, and train only on labels already released by that training cutoff. Historical realised weather can be used to diagnose sensitivities, but deployment evaluation must use forecast weather.

| Layer | Primary evidence |
|---|---|
| Weather | MAE/bias for temperature and humidity; event Brier score for rain; daylight irradiance error where observed; results by lead/hour/regime |
| Uncertainty | Quantile pinball loss, interval coverage and width; CRPS for a full predictive distribution; event reliability |
| Demand | MW MAE/RMSE, peak/ramp errors, performance relative to calendar and raw-weather baselines |
| Solar | Daylight-normalised error, energy bias and ramp-event precision/recall against measured truth |
| Prices | Calibration and precision/recall for spikes; quantile loss; performance across supply and price-rule regimes |
| Economics | Out-of-sample cost or P&L relative to a feasible baseline, with costs and risk limits; uncertainty intervals over day/week blocks |
| Operation | On-time publication, missing-input frequency, forecast age, collection completeness, memory and disk use |

For a flexible-load example, energy cost is:

`cost_SGD = sum(power_MW[t] * 0.5_hours * applicable_price_SGD_per_MWh[t])`

Use the same required daily energy and operational constraints for every strategy. Compare a fixed feasible schedule, a baseline forecast policy, and a weather-enhanced policy; a perfect-foresight schedule is only an upper bound. If WEP is unavailable, report an energy-component USEP experiment explicitly. For a real asset, include efficiency, switching/degradation costs, contract charges and settlement details as applicable.

Do not reward apparent model quality obtained by dropping difficult missing-data days. Retain failed/missed forecasts in operational statistics and apply a predeclared fallback policy to decision evaluation. A 90-day forward test is an initial check, not proof across both monsoons, changing capacity, or rare scarcity events. Seek at least 12 months of seasonally diverse evidence before a broad robustness claim.

## 6. Hardware fit and scope decision

The local Windows hardware inventory on 6 November 2026 found:

| Resource | Observed |
|---|---|
| CPU | Intel Core i7-1355U; 10 cores, 12 logical processors |
| Installed usable physical memory reported by Windows | 16,849,256,448 bytes, about 15.69 GiB |
| Free physical memory at inspection | About 2.00 GiB; a transient reading |
| Graphics | Intel Iris Xe integrated graphics; no NVIDIA device reported and no `nvidia-smi` found |
| Main disk | About 235.5 GiB total, 15.08 GiB free |
| OS | Windows 11 Business, 64-bit |

The video-controller memory field is not evidence of dedicated CUDA VRAM. No sustained CPU, network or training benchmark was run. Treat all performance limits in the plan as targets to measure.

This hardware can support compact station/point data, a DuckDB/Parquet archive, small CPU models, and a local report or dashboard. The immediate constraints are free storage, available RAM and reliable collection while the laptop sleeps. Global GRIB-message downloads can still be large even if only Singapore values are retained after decoding.

Start with a 2 GiB project-data cap, bounded regional/point downloads, one forecast job at a time and no full ensemble collection. Additional storage and an always-on collector are higher-priority upgrades than a GPU. An external 1 TB SSD and a modest always-on CPU machine would enable a durable archive; exact purchase or cloud specifications should follow a measured pilot.

The recommended scope is therefore **Singapore weather calibration and evaluation first; demand modelling second; verified solar and conditional price-risk work third**. Global model training, operational WRF/SINGV deployment, large satellite archives, multi-country market modelling, and automated trading are outside the initial project.
