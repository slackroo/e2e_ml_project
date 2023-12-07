import sys
import os
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException


def save_object(filepath,object):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,"wb") as file_obj:
            dill.dump(object,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

