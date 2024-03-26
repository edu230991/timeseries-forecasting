import warnings
import holidays
import inspect
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit


class NonLinearAutoRegressive:
    dataset_kwargs = [
        "linear_tree",
        "max_bin",
        "min_data_in_bin",
    ]
    dataset_params = None
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": 0,
        "min_child_samples": 1000,
        "num_leaves": 100,
        "num_boost_round": 500,
        "linear_tree": True,
        "boosting": "dart",
        "max_bin": 100,
    }

    def __init__(
        self,
        country: str,
        params: dict = {},
        context_lags: list = None,
        prediction_length: int = 1,
        prediction_freq: str = None,
    ):

        # check if any of the keywords belong to dataset for lgbm
        for key in self.dataset_kwargs:
            if key in params:
                if self.dataset_params is None:
                    self.dataset_params = {key: params.pop(key)}
                else:
                    self.dataset_params.update({key: params.pop(key)})

        # check if any of the params belong to the train function
        train_args = inspect.getfullargspec(lgb.train).args
        self.train_kwargs = {k: v for k, v in params.items() if k in train_args}
        [params.pop(k) for k in self.train_kwargs]

        self.params = self.default_params
        self.country = country
        self.context_lags = context_lags
        self.prediction_length = prediction_length
        self.prediction_freq = prediction_freq

    def validate_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # make a copy of the dataset to make sure not to alter the original
        dataset = dataset.copy()

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
        for j in range(self.prediction_length):
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

    def get_dataset(self, x: pd.DataFrame, y: pd.DataFrame = None):
        data = lgb.Dataset(
            x,
            label=y,
            params=self.dataset_params,
            free_raw_data=False,
        )
        return data

    def fit(self, dataset: pd.DataFrame, valid_share: float = 0.1):
        x, y = self.prepare_xy(dataset)
        self.dataset = dataset.copy()
        if valid_share > 0:
            x_val = x.sample(frac=valid_share, replace=False)
            y_val = y.reindex(x_val.index)
            x = x.drop(x_val.index, axis=0)
            y = y.reindex(x.index)
            valid_sets = [self.get_dataset(x_val, y_val)]
        else:
            valid_sets = []

        self.train_data = self.get_dataset(x, y)
        self.model = lgb.train(
            self.params, self.train_data, valid_sets=valid_sets, **self.train_kwargs
        )

    def prepare_x_pred(
        self, dataset: pd.DataFrame, cal_features: pd.DataFrame
    ) -> pd.DataFrame:

        x_pred = dataset.iloc[[-i - 1 for i in self.context_lags]].copy()
        x_pred = x_pred.T.sort_index(axis=1, ascending=False)
        x_pred.columns = pd.Index(self.context_lags, name="features")
        x_pred = self.repeat_df(x_pred, range(self.prediction_length), "horizon")
        cal_features = self.repeat_df(cal_features, self.targets, self.targets_name)

        x_pred = x_pred.stack("horizon", future_stack=True)
        x_pred = x_pred.join(
            cal_features.stack(self.targets_name, future_stack=True)
        ).reset_index()
        x_pred = self.ensure_types_and_order(x_pred)
        return x_pred

    def predict(self, steps: int = 1) -> pd.DataFrame:
        n_timesteps = steps * self.prediction_length
        dtrange = pd.date_range(
            start=self.dataset.index[-1] + self.timestep,
            end=self.dataset.index[-1] + self.timestep * n_timesteps,
            periods=n_timesteps,
        )
        cal_features = self.get_calendar_features(dtrange)

        preds = []
        for i in range(steps):
            this_cal_features = cal_features.iloc[
                i * self.prediction_length : (i + 1) * self.prediction_length
            ].reset_index(drop=True)
            this_cal_features.index.name = "horizon"

            x_pred = self.prepare_x_pred(self.dataset, this_cal_features)
            this_pred = (
                pd.Series(
                    self.model.predict(x_pred),
                    index=x_pred.set_index(["horizon", self.targets_name]).index,
                )
                .unstack()
                .sort_index()
            )
            this_pred.index = self.dataset.index[-1] + (
                self.timestep * (this_pred.index.astype(int) + 1)
            )
            preds.append(this_pred)
            self.dataset = pd.concat([self.dataset, this_pred])
        preds = pd.concat(preds).sort_index()
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

            train_data = self.get_dataset(x_train, y_train)
            test_data = self.get_dataset(x_test, y_test)
            self.model = lgb.train(
                self.params, train_data, valid_sets=[test_data], **self.train_kwargs
            )

            raw_test_data = test_data.get_data()
            self.y_pred = pd.Series(
                self.model.predict(raw_test_data), index=y_test.index
            )

            groupby = ["horizon", self.targets_name]

            rel_rmse = ((y_test - self.y_pred) ** 2).groupby(
                level=groupby, observed=False
            ).mean() ** 0.5 / y_test.abs().groupby(level=groupby, observed=False).mean()
            score[split] = rel_rmse
        print("")
        return pd.concat(score, axis=1, names=["fold"])
