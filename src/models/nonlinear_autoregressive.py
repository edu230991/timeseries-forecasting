import holidays
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator


class NonLinearAutoRegressive:
    params = {
        "verbosity": -1,
        "min_child_samples": 1000,
        "num_leaves": 100,
        "n_estimators": 500,
        "linear_tree": True,
        # "boosting_type": "dart",
        # "max_bin": 100,
    }
    model = lgb.LGBMRegressor

    def __init__(
        self,
        country: str,
        params: dict = {},
        context_lags: list = None,
        forecast_horizon: list = [1],
        prediction_freq: str = None,
        model: BaseEstimator = None,
    ):

        self.params.update(params)
        self.country = country
        self.context_lags = context_lags
        self.forecast_horizon = forecast_horizon
        self.prediction_freq = prediction_freq

        if model is not None:
            if isinstance(model, type):
                self.model = model(**self.params)
            elif isinstance(model, BaseEstimator):
                self.model = model
        else:
            self.model = self.model(**self.params)

    def validate_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # make a copy of the dataset to make sure not to alter the original
        dataset = dataset.copy()

        if isinstance(dataset.index, pd.MultiIndex) & (dataset.shape[1] == 1):
            # dataset is sktime style
            self.sktime_style = True
            dataset = dataset.iloc[:, 0].unstack(0)
        else:
            self.sktime_style = False

        # check that column list is correct
        if hasattr(self, "targets"):
            if self.targets != dataset.columns.tolist():
                raise ValueError(f"Target list should be {self.targets}")
        else:
            self.targets = dataset.columns.tolist()

        # name the column index
        if dataset.columns.name is not None:
            if hasattr(self, "targets_name"):
                if self.targets_name != dataset.columns.name:
                    raise ValueError(f"Column index name should be {self.targets_name}")
            else:
                self.targets_name = dataset.columns.name
        else:
            self.targets_name = "targets"
            dataset.columns.name = self.targets_name

        # name the index
        if dataset.index.name is not None:
            if hasattr(self, "index_name"):
                if self.index_name != dataset.index.name:
                    raise ValueError(f"Index name should be {self.targets_name}")
            else:
                self.index_name = dataset.index.name
        else:
            self.index_name = "datetime"
            dataset.index.name = self.index_name

        # detect dataset frequency
        if not hasattr(self, "timestep"):
            self.timestep = dataset.index[1] - dataset.index[0]
        else:
            if self.timestep != dataset.index[1] - dataset.index[0]:
                raise ValueError(f"Dataset time frequency should be {self.timestep}")

        self.categorical_features = [self.targets_name, "horizon"]
        return dataset

    def repeat_df(self, df: pd.DataFrame, labels: list, name: str):
        # repeat matrix n times
        dfdf = pd.concat({i: df for i in labels}, axis=1, names=[name])
        return dfdf

    def prepare_xy(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame]:
        dataset = self.validate_dataset(dataset)
        self.dataset = dataset.copy()
        cal_info = self.get_calendar_features(dataset.index)

        if self.prediction_freq:
            dts = pd.date_range(
                start=dataset.index[0].ceil(self.prediction_freq),
                end=dataset.index[-1].floor(self.prediction_freq),
                freq=self.prediction_freq,
                name=self.index_name,
            )
        else:
            dts = dataset.index
        x = {}
        for i in self.context_lags:
            x[i] = dataset.shift(i).dropna().reindex(dts).stack(future_stack=True)

        y = {}
        dataset.index = pd.MultiIndex.from_frame(cal_info.reset_index())
        for j in self.forecast_horizon:
            y[j] = (
                dataset.shift(-j)
                .dropna()
                .reindex(dts, level=0)
                .stack(future_stack=True)
            )

        x = pd.concat(x, axis=1, names=["features"]).dropna()
        y = pd.concat(y, axis=0, names=["horizon"]).dropna()

        x = x.reset_index()
        y = y.astype(float).to_frame("value").reset_index()
        x = x.merge(y, how="right", on=[self.index_name, self.targets_name]).dropna()

        y = x.set_index([self.index_name, self.targets_name, "horizon"])["value"]
        x = x.drop([self.index_name, "value"], axis=1)
        x.index = y.index
        x = self.ensure_types_and_order(x)

        return x.sort_index(), y.sort_index()

    def ensure_types_and_order(self, df: pd.DataFrame):
        float_cols = {k: float for k in self.context_lags}
        cat_cols = {k: "category" for k in self.categorical_features if k[:3] != "is_"}
        bool_cols = {k: bool for k in self.categorical_features if k[:3] == "is_"}
        df = df.astype({**float_cols, **cat_cols, **bool_cols})
        df = df.reindex(self.context_lags + self.categorical_features, axis=1)
        return df

    def get_calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        holiday_dict = holidays.country_holidays(self.country)
        hour = index.hour
        weekday = index.weekday
        is_holiday = pd.Series([d in holiday_dict for d in index.date], index=index)
        cal_df = pd.DataFrame(
            {
                "hour": hour.astype(int),
                "weekday": weekday.astype(int),
                "is_holiday": is_holiday.astype(bool),
            }
        )
        cal_df.columns.name = "features"
        self.calendar_features = cal_df.columns.tolist()
        if not all(
            [col in self.categorical_features for col in self.calendar_features]
        ):
            self.categorical_features += self.calendar_features
        return cal_df

    def fit(
        self, dataset: pd.DataFrame, fh: list = None, valid_share: float = 0, **kwargs
    ):
        if fh is not None:
            self.forecast_horizon = fh
        x, y = self.prepare_xy(dataset)
        if valid_share > 0:
            x_val = x.sample(frac=valid_share, replace=False)
            y_val = y.reindex(x_val.index)
            x = x.drop(x_val.index, axis=0)
            y = y.reindex(x.index)
            eval_set = [(x_val, y_val)]
        else:
            eval_set = []
        self.model.fit(x, y, eval_set=eval_set)

    def prepare_x_pred(self, dataset: pd.DataFrame) -> pd.DataFrame:
        x_pred = dataset.iloc[[-i - 1 for i in self.context_lags]].copy()
        x_pred = x_pred.T.sort_index(axis=1, ascending=False)
        x_pred.columns = pd.Index(self.context_lags, name="features")
        x_pred = self.repeat_df(x_pred, self.forecast_horizon, "horizon")
        x_pred = x_pred.stack("horizon", future_stack=True)
        x_pred[self.index_name] = (
            x_pred.index.get_level_values(1) * self.timestep + dataset.index[-1]
        )
        cal_features = self.get_calendar_features(
            pd.DatetimeIndex(x_pred[self.index_name])
        ).set_index(x_pred.index)
        x_pred = (
            pd.concat([x_pred, cal_features], axis=1)
            .reset_index()
            .set_index(self.index_name)
        )
        x_pred = self.ensure_types_and_order(x_pred)
        return x_pred

    def predict(self, fh: list = None) -> pd.DataFrame:
        if fh is None:
            fh = self.forecast_horizon
        if not isinstance(fh, pd.Series):
            fh = pd.Series(fh)

        dtrange = fh * self.timestep + self.dataset.index[-1]
        fh = fh.set_axis(dtrange)

        dataset = self.dataset.copy()
        preds = []
        while True:
            x_pred = self.prepare_x_pred(dataset)
            this_pred = (
                pd.Series(
                    self.model.predict(x_pred),
                    index=x_pred.set_index(self.targets_name, append=True).index,
                )
                .unstack()
                .sort_index()
            )
            preds.append(this_pred)

            if this_pred.index[-1] >= dtrange.max():
                break

            dataset = pd.concat([dataset, this_pred])
        preds = pd.concat(preds).sort_index().reindex(dtrange)
        if self.sktime_style:
            preds = preds.stack().swaplevel().to_frame("value").sort_index()
        return preds

    def fit_predict(self, dataset: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        self.fit(dataset)
        return self.predict(steps)

    def evaluate(
        self,
        dataset: pd.DataFrame,
        cv_splits: int,
        max_train_size: int = None,
        gap: int = 0,
        **kwargs,
    ) -> pd.DataFrame:

        tscv = TimeSeriesSplit(
            n_splits=cv_splits, max_train_size=max_train_size, gap=gap, **kwargs
        )
        x, y = self.prepare_xy(dataset)
        x, y = x.sort_index(), y.sort_index()
        dts = y.reset_index()[self.index_name].unique()

        score = {}
        for split, (train_index, test_index) in enumerate(tscv.split(dts)):
            print(
                f"Evaluating fold number {split+1}. "
                f"{len(train_index)} training timesteps and {len(test_index)} testing.",
                end="\r",
            )
            train_dts = dts[train_index]
            test_dts = dts[test_index]
            x_train = x.truncate(train_dts[0], train_dts[-1])
            x_test = x.truncate(test_dts[0], test_dts[-1])
            y_train = y.reindex(x_train.index)
            y_test = y.reindex(x_test.index)

            self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
            self.y_pred = pd.Series(self.model.predict(x_test), index=y_test.index)

            groupby = ["horizon", self.targets_name]

            rel_rmse = ((y_test - self.y_pred) ** 2).groupby(
                level=groupby, observed=False
            ).mean() ** 0.5 / y_test.abs().groupby(level=groupby, observed=False).mean()
            score[split] = rel_rmse
        print("")
        return pd.concat(score, axis=1, names=["fold"])
