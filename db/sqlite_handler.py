import sqlite3
from common.global_variables import *


class SqliteHandler:
    def __init__(self, sqlite3_db_filename):
        self.conn = sqlite3.connect(sqlite3_db_filename, timeout=10, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout = 3000")

    def create_tables(self, coin_names):
        cursor = self.conn.cursor()

        for coin_name in coin_names:
            ticker = "KRW_" + coin_name
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS {0} (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                datetime TEXT, open_price FLOAT, high_price FLOAT, low_price FLOAT, close_price FLOAT, volume FLOAT,
                total_ask_size FLOAT, total_bid_size FLOAT, btmi FLOAT, btmi_rate FLOAT, 
                btai FLOAT, btai_rate FLOAT)""".format(ticker))

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS BUY_SELL (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            coin_name TEXT, buy_datetime DATETIME, cnn_prob FLOAT, lstm_prob FLOAT, buy_price FLOAT, 
            trail_datetime DATETIME, trail_price FLOAT, trail_rate FLOAT, status TINYINT
            )""")

        self.conn.commit()

    def drop_tables(self, coin_names):
        cursor = self.conn.cursor()
        for coin_name in coin_names:
            ticker = "KRW_" + coin_name
            cursor.execute("DROP TABLE IF EXISTS {0}".format(ticker))

        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    sql_handler = SqliteHandler(sqlite3_db_filename)
    sql_handler.create_tables(UPBIT.get_all_coin_names())
