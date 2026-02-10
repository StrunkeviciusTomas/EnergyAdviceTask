import json
import click
import pandas as pd
import numpy as np
import base64
import requests

from eapy.io.template_io import load_task_data, dump_task_data
from eapy.io.managers.product_entry_manager import ProductEntryManager


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

class EASASEndpoint:
    def __init__(self, easas, ea_api_key):
        self.rest_root = f"{easas}/EASAS/rest"
        self.headers = {
            "Authorization": f"Bearer {ea_api_key}",
            "User-Agent": self._decode_username(ea_api_key),
        }

    def _decode_username(self, token):
        payload = token.split('.')[1]
        padding = '=' * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload + padding)
        return json.loads(decoded).get("username", "")


class RecentData(EASASEndpoint):
    def get_recent_data_df(self, identifiers: list, timestamp=None) -> pd.DataFrame:

        if timestamp is not None:
            timestamp = timestamp.isoformat()

        response = requests.post(
            f"{self.rest_root}/datasetData/dataInfoRecent",
            params={'refTime': timestamp} if timestamp else None,
            json=identifiers,
            headers=self.headers,
            timeout=(20, 70),
        )
        response.raise_for_status()

        raw = response.json()
        if not raw:
            return pd.DataFrame(columns=["time", "identifier", "value"])

        df = pd.json_normalize(
            raw,
            record_path=["data"],
            meta=["identifier"]
        ).rename(columns={"key": "time"})

        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        return df[["time", "identifier", "value"]]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_recent_value(group, dataset_name):
    ds = group.get_input_named_dataset(dataset_name)
    if ds is None:
        return np.nan

    s = ds.get_as_series().dropna()
    if s.empty:
        return np.nan

    # Assumes time-ordered series (standard in EAPY)
    return int(s.iloc[-1])


def get_control_groups_df(group):
    em = ProductEntryManager(group.templates)
    df = em.get_all_entries_as_pandas_df("control_groups_table")

    df["mapping"] = pd.to_numeric(df["mapping"], errors="coerce")

    return df[[
        "supply_system",
        "mapping",
        "worst_room_name_ds",
        "worst_room_dt_ds"
    ]]


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

EASAS = "i will change"
EA_API_KEY = "i will change"

my_inputs = {
    "controlled_by_m1_13": "M1 NH₃ -13°C",
    "controlled_by_m1_42": "M1 NH₃ -42°C",
    "controlled_by_m3_ammonia": "M3 NH₃ -13°C",
}

my_outputs = {
    "controlled_by_m1_13": "worst_room_m1_13",
    "controlled_by_m1_42": "worst_room_m1_42",
    "controlled_by_m3_ammonia": "worst_room_m3_ammonia",
}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

@click.command()
@click.option("--input_file_path", default="input.json")
@click.option("--output_file_path", default="output.json")
def main(input_file_path, output_file_path):

    with open(input_file_path, encoding="utf-8") as f:
        json_input = json.load(f)

    input_data = load_task_data(json_input)
    gr = input_data.get_default_data_group(key="data")

    recent_data_getter = RecentData(EASAS, EA_API_KEY)
    table_df = get_control_groups_df(gr)

    ts = pd.Timestamp.now(tz="Europe/Vilnius")

    for input_name, system_name in my_inputs.items():

        # 1️⃣ Most recent mapping value from JSON
        mapping_value = get_recent_value(gr, input_name)

        # 2️⃣ Filter by supply_system AND mapping
        row = table_df[
            (table_df["supply_system"] == system_name) &
            (table_df["mapping"] == mapping_value)
        ].iloc[0]

        worst_name_ds = row["worst_room_name_ds"]
        worst_dt_ds = row["worst_room_dt_ds"]

        # 3️⃣ Call API for dataset IDs
        identifiers = [worst_name_ds, worst_dt_ds]
        df_recent = recent_data_getter.get_recent_data_df(identifiers, ts)

        # 4️⃣ Take most recent value per identifier
        df_last = (
            df_recent
            .sort_values("time")
            .groupby("identifier")
            .tail(1)
        )

        value_lookup = dict(zip(df_last["identifier"], df_last["value"]))

        worst_name_value = value_lookup.get(worst_name_ds)
        worst_dt_value = value_lookup.get(worst_dt_ds)

        # 5️⃣ Output only numeric values
        out_df = pd.DataFrame([{
            "mapping_value": mapping_value,
            "worst_room_name_value": worst_name_value,
            "worst_room_dt_value": worst_dt_value
        }])

        output_dataset_name = my_outputs[input_name]
        gr.get_output_named_dataset(output_dataset_name).set_from_dataframe(out_df)

    # Save output
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            out_data = input_data.get_as_output_data()
            json.dump(dump_task_data(out_data), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
