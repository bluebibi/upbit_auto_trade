import math
import sys, os
import time
from datetime import datetime, timedelta, date
import pickle
import numpy as np

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)


def convert_unit_2(unit):
    if not isinstance(unit, float):
        unit = float(unit)
    converted_unit = math.floor(unit * 100) / 100
    return converted_unit
