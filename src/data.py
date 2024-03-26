import os
import pandas as pd
from pyenerginet import EnerginetData
from src.config import DATA_DIR


def get_danish_industry_consumption(
    start: pd.Timestamp = None,
    end: pd.Timestamp = None,
    from_file: bool = True,
    to_file: bool = False,
    size: int = None,
):
    filepath = os.path.join(DATA_DIR, "dk-ind-cons.pkl")
    if from_file:
        df = pd.read_pickle(filepath)
    else:
        if end is None:
            end = pd.Timestamp("today", tz="CET").normalize() - pd.offsets.MonthBegin()
        if start is None:
            start = end - pd.to_timedelta(730, unit="d")
        endk = EnerginetData()
        df = endk.get_consumption_per_industry_code(start, end)
        df = df.pivot(columns="DK36Title", values="Consumption_MWh")
        if to_file:
            df.to_pickle(filepath)
    if size is not None:
        df = df.iloc[:-size]
    return df
