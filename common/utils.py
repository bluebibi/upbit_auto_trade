import math
import sys, os
import time
from datetime import datetime, timedelta, date
import pickle
import numpy as np

from common.global_variables import CoinStatus

idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)


def convert_unit_2(unit):
    if unit:
        if not isinstance(unit, float):
            unit = float(unit)
        converted_unit = math.floor(unit * 100) / 100
        return converted_unit
    else:
        return unit


def convert_unit_4(unit):
    if unit:
        if not isinstance(unit, float):
            unit = float(unit)
        converted_unit = math.floor(unit * 10000) / 10000
        return converted_unit
    else:
        return unit


def coin_status_to_hangul(status):
    if status == CoinStatus.bought.value:
        status = "구매"
    elif status == CoinStatus.trailed.value:
        status = "추적"
    elif status == CoinStatus.success_sold.value:
        status = "성공 매도"
    elif status == CoinStatus.gain_sold.value:
        status = "이득 매도"
    elif status == CoinStatus.loss_sold.value:
        status = "손실 매도"

    return status
