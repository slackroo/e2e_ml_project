import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging, project_root
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(project_root, "artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """
         function used to transform the data
        :return: preprocessed data
        """
        try:
            num_cols = ['writing_score', 'reading_score']
            cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])
            logging.info(f"num pipeline with std scaling performed for {num_cols} ")
            cat_pipeline = Pipeline([
                ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot', OneHotEncoder())

            ])
            logging.info(f"cat pipeline with one hot encoding performed for {cat_cols} ")

            full_pipeline = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols),
            ])
            return full_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the train data completed")
            logging.info('Obtain full pipeline')
            pipeline_obj = self.get_data_transformer_obj()

            target_column = 'math_score'
            num_cols = ['writing_score', 'reading_score']

            train_set = train_df.drop(target_column, axis=1)
            test_set = test_df.drop(target_column, axis=1)
            train_labels = train_df[target_column].copy()
            test_labels = test_df[target_column].copy()

            logging.info(f"Applying the full pipeline on training data and test set")

            train_prepared = pipeline_obj.fit_transform(train_set)
            test_prepared = pipeline_obj.transform(test_set)

            train_arr = np.c_[
                train_prepared, np.array(train_labels)
            ]
            test_arr = np.c_[test_prepared, np.array(test_labels)]

            logging.info("save the pipeline pkl file")

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                object=pipeline_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
