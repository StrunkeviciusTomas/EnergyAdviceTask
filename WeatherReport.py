from dataclasses import dataclass
from typing import Any, Optional
from datetime import date, datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt

LT_TZ = "Europe/Vilnius"


def interpolate_to_5min(temperature: pd.Series) -> pd.Series:
    if not isinstance(temperature.index, pd.DatetimeIndex):
        raise TypeError("temperature must have a DatetimeIndex")
    if temperature.index.tz is None:
        raise ValueError("temperature index must be tz-aware")

    s = temperature.sort_index()
    s5 = s.resample("5min").asfreq()
    return s5.interpolate(method="time")


def _weekend_id_lt(ts: pd.Timestamp) -> pd.Timestamp:
    wd = ts.weekday()
    if wd == 5:
        return ts.normalize()
    if wd == 6:
        return (ts - pd.Timedelta(days=1)).normalize()
    return pd.NaT


@dataclass(frozen=True)
class MeteoLTClient:
    place_code: str
    base_url: str = "https://api.meteo.lt/v1"
    tz: str = LT_TZ
    station_code: Optional[str] = None
    timeout_s: int = 30

    def _get_json(self, path: str) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        with requests.Session() as s:
            s.headers.update({"Accept": "application/json"})
            r = s.get(url, timeout=self.timeout_s)
            r.raise_for_status()
            return r.json()

    def _place_coordinates(self) -> tuple[float, float]:
        payload = self._get_json(f"/places/{self.place_code}")
        c = payload["coordinates"]
        return float(c["latitude"]), float(c["longitude"])

    def _stations(self) -> list[dict]:
        payload = self._get_json("/stations")
        if isinstance(payload, list):
            return payload
        return payload.get("stations", [])

    def _infer_nearest_station(self) -> str:
        lat0, lon0 = self._place_coordinates()
        stations = self._stations()

        def dist2(st: dict) -> float:
            c = st["coordinates"]
            return (c["latitude"] - lat0) ** 2 + (c["longitude"] - lon0) ** 2

        return min(stations, key=dist2)["code"]

    def get_station_code(self) -> str:
        return self.station_code or self._infer_nearest_station()

    def read_historical(self, start: date | datetime, end: date | datetime) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        start_ts = start_ts.tz_localize(self.tz) if start_ts.tzinfo is None else start_ts.tz_convert(self.tz)
        end_ts = end_ts.tz_localize(self.tz) if end_ts.tzinfo is None else end_ts.tz_convert(self.tz)

        utc_days = pd.date_range(
            start_ts.tz_convert("UTC").normalize(),
            end_ts.tz_convert("UTC").normalize(),
            freq="D",
        )

        station = self.get_station_code()
        rows: list[dict] = []

        total = len(utc_days)
        for i, d in enumerate(utc_days, 1):
            print(f"\rFetching data: {i}/{total} days", end="", flush=True)
            payload = self._get_json(f"/stations/{station}/observations/{d:%Y-%m-%d}")
            rows.extend(payload.get("observations", []))

        print()

        if not rows:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz=self.tz))

        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["observationTimeUtc"], utc=True).dt.tz_convert(self.tz)
        df = df.drop(columns="observationTimeUtc").set_index(idx).sort_index()
        df.index.name = "time"
        return df

    def read_forecast(self) -> pd.DataFrame:
        payload = self._get_json(f"/places/{self.place_code}/forecasts/long-term")
        df = pd.DataFrame(payload["forecastTimestamps"])
        idx = pd.to_datetime(df["forecastTimeUtc"], utc=True).dt.tz_convert(self.tz)
        df = df.drop(columns="forecastTimeUtc").set_index(idx).sort_index()
        df.index.name = "time"
        return df


def compute_year_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    hours = hist.index.hour
    day_mask = (hours >= 8) & (hours < 20)

    rain = pd.Series(False, index=hist.index)
    if "precipitation" in hist.columns:
        rain |= hist["precipitation"].fillna(0) > 0
    if "conditionCode" in hist.columns:
        rain |= hist["conditionCode"].str.contains("rain|thunder", case=False, na=False)

    weekend_ids = hist.index.to_series().apply(_weekend_id_lt)
    weekends_with_rain = int(rain.groupby(weekend_ids).any().sum())

    return pd.DataFrame(
        {
            "value": [
                hist["airTemperature"].mean(),
                hist["relativeHumidity"].mean(),
                hist.loc[day_mask, "airTemperature"].mean(),
                hist.loc[~day_mask, "airTemperature"].mean(),
                weekends_with_rain,
            ]
        },
        index=[
            "avg_temperature_c",
            "avg_relative_humidity_pct",
            "avg_day_temperature_c (08-20 LT)",
            "avg_night_temperature_c",
            "weekends_with_rain_count (measured)",
        ],
    )


def plot_last_week(hist: pd.DataFrame, forecast: pd.DataFrame) -> None:
    now = pd.Timestamp.now(tz=hist.index.tz)

    hist_last = hist.loc[hist.index >= now - pd.Timedelta(days=7), "airTemperature"]
    fc_future = forecast.loc[forecast.index >= now, "airTemperature"]

    plt.figure()
    hist_last.plot(label="Measured (last week)")
    fc_future.plot(label="Forecast")
    plt.legend()
    plt.xlabel("Time (LT)")
    plt.ylabel("Temperature (°C)")
    plt.title("Measured vs forecast temperature")
    plt.tight_layout()
    plt.savefig("temperature_kaunas.png")
    plt.show()


def run_analysis(place_code: str = "kaunas") -> dict[str, Any]:
    client = MeteoLTClient(place_code=place_code)

    end = pd.Timestamp.now(tz=LT_TZ)
    start = end - pd.Timedelta(days=365)

    historical = client.read_historical(start, end)
    forecast = client.read_forecast()
    metrics = compute_year_metrics(historical)

    return {
        "historical_df": historical,
        "forecast_df": forecast,
        "metrics_df": metrics,
        "station": client.get_station_code(),
    }


if __name__ == "__main__":
    print("Running weather analysis...\n")

    results = run_analysis("kaunas")

    print("Station used:", results["station"])
    print("\nMetrics:")
    print(results["metrics_df"])

    print("\nHistorical data sample:")
    print(results["historical_df"].head())

    print("\nForecast data sample:")
    print(results["forecast_df"].head())

    plot_last_week(results["historical_df"], results["forecast_df"])

    temp_5m = interpolate_to_5min(results["historical_df"]["airTemperature"].dropna().tail(24))
    print("\nInterpolated 5-minute temperature sample:")
    print(temp_5m.head(10))
