import pandas as pd
from matplotlib import pyplot as plt

from src.models import *
from src.data import get_danish_industry_consumption

df = get_danish_industry_consumption()
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 100,
    "min_child_samples": 50,
    "num_boost_round": 100,
    "learning_rate": 0.1,
    "verbosity": -1,
    "linear_tree": True,
}
for cl in [168, 336]:
    model = NonLinearAutoRegressive(context_length=cl, lag=48, params=params)
    # print(params)
    score = model.evaluate(df, cv_splits=5, max_train_size=10000, gap=48)
    weighted_score = (
        score.mean(axis=1) * df.abs().mean()
    ).sum() / df.abs().mean().sum()
    print(weighted_score)

# model.fit(df.iloc[:-72])
# pred = model.predict(df.iloc[:-72], steps=72)

# ax = pred.plot(linestyle="--", legend=False)
# df.tail(200).plot(ax=ax, legend=False)

# import ipdb

# ipdb.set_trace()
