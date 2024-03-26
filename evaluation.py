import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.compose import make_reduction

from src.models import *
from src.data import get_danish_industry_consumption
from tqdm import tqdm

os.environ["LOKY_MAX_CPU_COUNT"] = "2"


def evaluate_nlar(df: pd.DataFrame):
    prediction_length = 24
    context_lags = list(range(48, 73)) + list(range(168, 168 + 25))

    model = NonLinearAutoRegressive(
        context_lags=context_lags,
        country="Denmark",
        prediction_length=prediction_length,
        prediction_freq="d",
    )

    score = model.evaluate(df, cv_splits=3)
    print(score.mean().mean())

    print("Fitting model")
    model.fit(df)
    print("Predicting")
    pred = model.predict(steps=3)

    ax = pred.plot(linestyle="--", legend=False)
    df.tail(200).plot(ax=ax, legend=False)
    plt.show()


df = get_danish_industry_consumption()
df_sktime = (
    df.tz_convert("UTC").tz_localize(None).asfreq("H").stack().to_frame("value")
).sort_index()

tscv = TimeSeriesSplit(n_splits=365, max_train_size=10000, test_size=24)
dts = df_sktime.index.get_level_values(0).unique()

forecasters = {
    "knn_compose_multi": make_reduction(
        KNeighborsRegressor(n_neighbors=100), window_length=168, strategy="multioutput"
    ),
    "knn_compose_rec": make_reduction(
        KNeighborsRegressor(n_neighbors=100), window_length=168, strategy="recursive"
    ),
    "rf_compose_multi": make_reduction(
        RandomForestRegressor(n_estimators=100, min_samples_leaf=100),
        window_length=168,
        strategy="multioutput",
    ),
    "exp_smooth": ExponentialSmoothing(
        seasonal="add", sp=168, use_boxcox=True, use_brute=False
    ),
    "tbats": TBATS(
        use_box_cox=True, euse_trend=False, sp=[168, 24], show_warnings=False
    ),
}


prediction_horizon = range(48, 48 + 25)

preds = []
for split, (train_index, test_index) in enumerate(tscv.split(dts)):
    train_dts = dts[train_index]
    test_dts = dts[test_index]
    x_train = df_sktime.truncate(train_dts[0], train_dts[-1])
    x_test = df_sktime.truncate(test_dts[0], test_dts[-1])

    for name, forecaster in forecasters.items():
        print(f"--------- Training {name} on split {split+1}", end="\r")
        forecaster.fit(x_train.swaplevel(), fh=prediction_horizon)
        y_pred = forecaster.predict(prediction_horizon)["value"].to_frame(name)
        preds.append(y_pred)
    import ipdb

    ipdb.set_trace()
