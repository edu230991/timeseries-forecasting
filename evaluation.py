import pandas as pd
from matplotlib import pyplot as plt

from src.models import *
from src.data import get_danish_industry_consumption

df = get_danish_industry_consumption()[["Privat"]]
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 100,
    "min_child_samples": 100,
    "num_boost_round": 10,
    "learning_rate": 0.1,
    "verbosity": -1,
    "linear_tree": True,
}
prediction_length = 6
context_lags = list(range(48, 72)) + list(range(168, 168 + 24))
model = NonLinearAutoRegressive(
    context_lags=context_lags,
    country="Denmark",
    prediction_length=prediction_length,
    params=params,
)
# score = model.evaluate(df, cv_splits=4, max_train_size=10000, gap=48)
# print("Score:", round(score.mean().mean(), 2))

print("Fitting model")
model.fit(df)
print("Predicting")
pred = model.predict(df, steps=4)

ax = pred.plot(linestyle="--", legend=False)
df.tail(200).plot(ax=ax, legend=False)
plt.show()

import ipdb

ipdb.set_trace()

# x, y = model.prepare_xy(df)
# dataset = model.get_stacked_dataset(x, y)


# # print(params)
# score = model.evaluate(df, cv_splits=5, max_train_size=10000, gap=48)
# weighted_score = (
#     score.mean(axis=1) * df.abs().mean()
# ).sum() / df.abs().mean().sum()
# print(weighted_score)

# model.fit(df.iloc[:-72])
# pred = model.predict(df.iloc[:-72], steps=72)


# import ipdb

# ipdb.set_trace()
