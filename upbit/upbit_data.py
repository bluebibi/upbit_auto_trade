from conf.config import *
from db.sqlite_handler import SqliteHandler
from upbit.upbit_api import Upbit
import pandas as pd

select_by_datetime = "SELECT * FROM {0};"

class Upbit_Data:
    def __init__(self, coin_name):
        self.sql_handler = SqliteHandler(sqlite3_db_filename)
        self.upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT)
        self.coin_name = coin_name

    def get_data(self, windows_size=10, future_target=10):
        self.cursor = self.sql_handler.conn.cursor()
        df = pd.read_sql_query(select_by_datetime.format("KRW_" + self.coin_name), self.sql_handler.conn)

        print(df)

if __name__ == "__main__":
    upbit_data = Upbit_Data('BTC')
    upbit_data.get_data()
