import sys
import os
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV


def save_object(filepath, object):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        # X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            train_model_r2_score = r2_score(y_train, train_preds)
            test_model_r2_score = r2_score(y_test, test_preds)
            train_model_rmse_score = np.sqrt(mean_squared_error(y_train, train_preds))
            test_model_rmse_score = np.sqrt(mean_squared_error(y_test, test_preds))
            report[list(models.keys())[i]] = test_model_r2_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)