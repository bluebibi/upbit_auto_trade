import sqlite3

import sys, os
idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(os.getcwd())

from common.global_variables import *

class SqliteHandler:
    def __init__(self, sqlite3_price_info_db_filename):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            conn.execute("PRAGMA busy_timeout = 3000")

    def create_tables(self, coin_names):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()

            for coin_name in coin_names:
                ticker = "KRW_" + coin_name
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS {0} (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    datetime TEXT, open_price FLOAT, high_price FLOAT, low_price FLOAT, close_price FLOAT, volume FLOAT,
                    total_ask_size FLOAT, total_bid_size FLOAT, btmi FLOAT, btmi_rate FLOAT, 
                    btai FLOAT, btai_rate FLOAT)""".format(ticker))

            conn.commit()

    def create_buy_sell_table(self):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS BUY_SELL (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                coin_ticker_name TEXT, buy_datetime DATETIME, cnn_prob FLOAT, lstm_prob FLOAT, buy_base_price FLOAT,
                buy_krw INT, buy_fee INT, buy_price FLOAT, buy_coin_volume FLOAT, 
                trail_datetime DATETIME, trail_price FLOAT, sell_fee INT, sell_krw INT, trail_rate FLOAT, 
                total_krw INT, status TINYINT
                )""")

            conn.commit()

    def drop_tables(self, coin_names):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            for coin_name in coin_names:
                ticker = "KRW_" + coin_name
                cursor.execute("DROP TABLE IF EXISTS {0}".format(ticker))

            conn.commit()


if __name__ == "__main__":
    sql_handler = SqliteHandler(sqlite3_price_info_db_filename)
    #sql_handler.create_tables(UPBIT.get_all_coin_names())
    sql_handler.create_buy_sell_table()
