import os
from enum import Enum
import configparser
import torch

from db.sqlite_handler import SqliteHandler
from upbit.slack import PushSlack
from upbit.upbit_api import Upbit


class CoinStatus(Enum):
    null = -1
    predicted = 0
    readyset = 1
    up_trailed = 2
    down_trailed = 3
    partial_sold = 4
    sold = 5
    ignored = 6


class Period(Enum):
    daily = 0
    half_daily = 1
    quater_daily = 2
    every_hour = 3


class BuyType(Enum):
    normal = 0
    prompt = 1


# GENERAL
fmt = "%Y-%m-%dT%H:%M:%S"
idx = os.getcwd().index("upbit_auto_trade")
BASE_DIR = os.getcwd()[:idx] + "upbit_auto_trade/"
sqlite3_db_filename = os.path.join(BASE_DIR, 'db/upbit_price_info.db')
order_book_info_filename = os.path.join(BASE_DIR, 'models/order_book_info.pickle')

config = configparser.ConfigParser()
read_ok = config.read(os.getcwd()[:idx] + "upbit_auto_trade/common/config.ini")

# USER
USER_ID = int(config['USER']['user_id'])
USERNAME = config['USER']['username']
HOST_IP = config['USER']['host_ip']
SYSTEM_USERNAME = config['USER']['system_username']
SYSTEM_PASSWORD = config['USER']['system_password']
EXCHANGE = config['USER']['exchange']
SOURCE = config['USER']['source']

# UPBIT
CLIENT_ID_UPBIT = config['UPBIT']['access_key']
CLIENT_SECRET_UPBIT = config['UPBIT']['secret_key']
FEE_UPBIT = 0.0005

#TELEGRAM
TELEGRAM_API_ID = config['TELEGRAM']['api_id']
TELEGRAM_API_HASH = config['TELEGRAM']['api_hash']
TELEGRAM_APP_TITLE = config['TELEGRAM']['app_title']

#SLACK
SLACK_WEBHOOK_URL_1 = config['SLACK']['webhook_url_1']
SLACK_WEBHOOK_URL_2 = config['SLACK']['webhook_url_2']

#TRAIN
USE_CNN_MODEL = config.getboolean('TRAIN', 'use_cnn_model')
NUM_EPOCHS = int(config['TRAIN']['num_epochs'])

#DATA
WINDOW_SIZE = int(config['DATA']['window_size'])
FUTURE_TARGET_SIZE = int(config['DATA']['future_target_size'])
UP_RATE = float(config['DATA']['up_rate'])
TRAIN_COLS_FULL = config.getboolean('DATA', 'train_cols_full')
if TRAIN_COLS_FULL:
    TRAIN_COLS = ["open_price", "high_price", "low_price", "close_price", "volume", "total_ask_size",
              "total_bid_size", "btmi", "btmi_rate", "btai", "btai_rate"]
else:
    TRAIN_COLS = ["open_price", "high_price", "low_price", "close_price", "volume"]
INPUT_SIZE = len(TRAIN_COLS)

VERBOSE = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPBIT = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)
SQL_HANDLER = SqliteHandler(sqlite3_db_filename)
SLACK = PushSlack(SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2)

if __name__ == "__main__":
    SQL_HANDLER.create_tables(UPBIT.get_all_coin_names())
    #SQL_HANDLER.drop_tables(UPBIT.get_all_coin_names())
