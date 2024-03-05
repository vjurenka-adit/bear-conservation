import os
from pathlib import Path

import pandas as pd
import yaml


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def resize_dataframe(df: pd.DataFrame, threshold_value: int):
    return df.groupby("bear_id").filter(lambda x: len(x) > threshold_value)


def list_subdirs(dir: Path):
    """Lists all subdirs in dir."""
    return [item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item))]


THRESHOLDS = {
    "nano": 150,
    "small": 100,
    "medium": 50,
    "large": 10,
    "xlarge": 1,
    "full": 0,
}
ALLOWED_ORIGINS = ["brooksFalls", "britishColumbia"]
