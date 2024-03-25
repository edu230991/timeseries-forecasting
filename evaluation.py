import itertools
import pandas as pd
from matplotlib import pyplot as plt

from src.models import *
from src.data import get_danish_industry_consumption

df = get_danish_industry_consumption()[["Privat"]]
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 50,
    "min_child_samples": 1000,
    "max_depth": 10,
    "num_boost_round": 200,
    "learning_rate": 0.1,
    "verbosity": -1,
    "linear_tree": True,
}

prediction_length = 24
context_lags = list(range(48, 73)) + list(range(168, 168 + 25))

model = NonLinearAutoRegressive(
    context_lags=context_lags,
    country="Denmark",
    prediction_length=prediction_length,
    params=params,
)

print("Fitting model")
model.fit(df)
print("Predicting")
pred = model.predict(steps=3)

ax = pred.plot(linestyle="--", legend=False)
df.tail(200).plot(ax=ax, legend=False)
plt.show()

# params_grid = {
#     "num_leaves": [50, 200, 1000],
#     "min_child_samples": [50, 100, 1000],
#     "max_depth": [-1, 5, 10],
#     "num_boost_round": [100, 200],
# }
# scores = pd.DataFrame()
# for values in itertools.product(*params_grid.values()):
#     new_params = params.copy()
#     point = dict(zip(params_grid.keys(), values))
#     new_params.update(point)
#     model = NonLinearAutoRegressive(
#         context_lags=context_lags,
#         country="Denmark",
#         prediction_length=prediction_length,
#         params=new_params,
#     )

#     score = model.evaluate(df, cv_splits=3, max_train_size=10000, gap=48)
#     point["score"] = round(score.mean().mean(), 4)
#     scores = (
#         pd.concat((scores, pd.Series(point).to_frame().T))
#         .sort_values("scores")
#         .reset_index()
#     )
#     print(scores)

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
