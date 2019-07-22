from conf.config import *
from db.sqlite_handler import SqliteHandler
from upbit.slack import PushSlack
from upbit.upbit_api import Upbit

price_insert = "INSERT INTO {0} VALUES(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
select_by_datetime = "SELECT * FROM {0} WHERE datetime='{1}';"


class Upbit_Recorder:
    def __init__(self):
        self.sql_handler = SqliteHandler(sqlite3_db_filename)
        self.upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT)
        self.coin_names = self.upbit.get_all_coin_names()

    def record(self, coin_name):
        i = self.upbit.get_market_index()

        ticker = "KRW-" + coin_name
        r = self.upbit.get_ohlcv(ticker, interval="minute10").values

        new_records = 0

        for row in r:
            datetime = row[0].replace('T', ' ')
            open_price = row[1]
            high_price = row[2]
            low_price = row[3]
            close_price = row[4]
            volume = row[5]

            if not self.exist_row_by_datetime(coin_name, datetime):
                o = self.upbit.get_orderbook(tickers=ticker)

                self.cursor = self.sql_handler.conn.cursor()

                self.cursor.execute(price_insert.format("KRW_" + coin_name), (
                    datetime,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    o[0]['total_ask_size'],
                    o[0]['total_bid_size'],
                    i['data']['btmi']['market_index'],
                    i['data']['btmi']['rate'],
                    i['data']['btai']['market_index'],
                    i['data']['btai']['rate']
                ))
                self.sql_handler.conn.commit()
                new_records += 1
        return new_records

    def exist_row_by_datetime(self, coin_name, datetime):
        self.cursor = self.sql_handler.conn.cursor()
        self.cursor.execute(select_by_datetime.format("KRW_" + coin_name, datetime))

        row = self.cursor.fetchall()
        if len(row) == 0:
            self.sql_handler.conn.commit()
            return False
        else:
            self.sql_handler.conn.commit()
            return True


if __name__ == "__main__":
    upbit_recorder = Upbit_Recorder()
    slack = PushSlack()

    total_new_records = 0
    for coin_name in upbit_recorder.coin_names:
        total_new_records += upbit_recorder.record(coin_name)

    msg = "Number of new upbit records: {0} @ {1}".format(total_new_records, SOURCE)
    slack.send_message("me", msg)



