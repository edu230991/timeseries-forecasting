import warnings
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit


class NonLinearAutoRegressive:
    # TODO
    # include other categorical features (holidays, weekday)
    # enable multi-output instead of looping. can turn it into single-output by using cat features

    def __init__(
        self,
        params: dict,
        context_length: int,
        lag: int = 0,
        prediction_length: int = 1,
    ):
        self.params = params

        if "linear_tree" in self.params:
            self.dataset_params = {"linear_tree": self.params.pop("linear_tree")}
        else:
            self.dataset_params = None

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.lag = lag

    def prepare_xy(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame]:
        if not hasattr(self, "timestep"):
            self.timestep = dataset.index[1] - dataset.index[0]
        x = {}
        for i in range(self.context_length):
            x[i + self.lag] = dataset.shift(i + 1 + self.lag).dropna()
        x = pd.concat(x, axis=1).dropna()
        y = dataset.reindex(x.index)
        return x, y

    def get_stacked_dataset(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self.cat_feature_name = (
            y.columns.name if y.columns.name is not None else "index"
        )

        x = x.stack(future_stack=True).reset_index(level=1)
        x[self.cat_feature_name] = x[self.cat_feature_name].astype("category")
        if y is not None:
            y = y.stack(future_stack=True)
        data = lgb.Dataset(
            x,
            label=y,
            params=self.dataset_params,
            categorical_feature=[self.cat_feature_name],
            free_raw_data=False,
        )
        return data

    def fit(self, dataset: pd.DataFrame):
        self.x, self.y = self.prepare_xy(dataset)
        self.train_data = self.get_stacked_dataset(self.x, self.y)
        with warnings.catch_warnings(action="ignore"):
            self.model = lgb.train(self.params, self.train_data)

    def predict(self, dataset: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        # make a copy of the dataset to make sure not to alter the original
        dataset = dataset.copy()

        preds = {}
        for _ in range(steps):
            dt = dataset.index[-1] + self.timestep
            x_pred = dataset.iloc[-self.context_length :].copy()
            x_pred = x_pred.T.sort_index(axis=1, ascending=False)
            x_pred.columns = range(self.lag, self.lag + self.context_length)
            x_pred = x_pred.reset_index()
            x_pred[self.cat_feature_name] = x_pred[self.cat_feature_name].astype(
                "category"
            )
            preds[dt] = pd.Series(
                self.model.predict(x_pred), index=x_pred[self.cat_feature_name]
            )
            dataset.loc[dt] = preds[dt].values
        preds = pd.concat(preds, axis=1).T
        return preds

    def fit_predict(self, dataset: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        self.fit(dataset)
        return self.predict(dataset, steps)

    def evaluate(
        self,
        dataset: pd.DataFrame,
        cv_splits: int,
        max_train_size: int,
        gap: int,
        **kwargs,
    ) -> pd.DataFrame:
        tscv = TimeSeriesSplit(
            n_splits=cv_splits, max_train_size=max_train_size, gap=gap, **kwargs
        )
        x, y = self.prepare_xy(dataset)

        score = pd.DataFrame()
        for split, (train_index, test_index) in enumerate(tscv.split(x)):
            print(
                f"Evaluating fold number {split+1}. "
                f"{len(train_index)} training rows and {len(test_index)} testing.",
                end="\r",
            )
            x_train = x.iloc[train_index].copy()
            x_test = x.iloc[test_index].copy()
            y_train = y.reindex(x_train.index)
            y_test = y.reindex(x_test.index)

            train_data = self.get_stacked_dataset(x_train, y_train)
            test_data = self.get_stacked_dataset(x_test, y_test)
            with warnings.catch_warnings(action="ignore"):
                self.model = lgb.train(self.params, train_data, valid_sets=[test_data])

            raw_test_data = test_data.get_data()
            test_index = raw_test_data.set_index(
                train_data.categorical_feature, append=True
            ).index
            self.y_pred = pd.Series(
                self.model.predict(raw_test_data), index=test_index
            ).unstack()

            rel_rmse = ((y_test - self.y_pred) ** 2).mean() ** 0.5 / y_test.abs().mean()
            score[split] = rel_rmse
        print("")
        return score
