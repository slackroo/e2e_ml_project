import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging, project_root
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join(project_root, 'artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test imput")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN regressor": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "Catboost": CatBoostRegressor(),
                "ADAboost": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models)

            # best r2 model
            best_model_r2_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_r2_score)]
            best_model = models[best_model_name]

            if best_model_r2_score < 0.6:
                raise CustomException("no best model found")

            logging.info(f"Best model found based on testing data ")

            save_object(
                filepath=self.model_trainer_config.trained_model_path,
                object=best_model
            )

            prediction = best_model.predict(X_test)
            r2_score_ = r2_score(y_test, prediction)
            rmse_score = np.sqrt(mean_squared_error(y_test,prediction))
            logging.info(f'Best model r2 score on the test set {best_model} with score of {r2_score_} and rmse of {rmse_score}')
            return r2_score_

        except Exception as e:
            raise CustomException(e, sys)
