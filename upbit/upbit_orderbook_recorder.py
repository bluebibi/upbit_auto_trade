import sqlite3
from pytz import timezone
import datetime as dt
import time

import sys, os
idx = os.getcwd().index("upbit_auto_trade")
PROJECT_HOME = os.getcwd()[:idx] + "upbit_auto_trade/"
sys.path.append(PROJECT_HOME)

from common.global_variables import *
from common.logger import get_logger


logger = get_logger("upbit_order_book_recorder_logger")

if os.getcwd().endswith("upbit"):
    os.chdir("..")

order_book_insert_sql = "INSERT INTO KRW_{0}_ORDER_BOOK VALUES(NULL, "\
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?," \
                        "?, ?, ?, ?, ?);"

select_by_datetime = "SELECT * FROM KRW_{0}_ORDER_BOOK WHERE base_datetime='{1}';"


class UpbitOrderBookRecorder:
    def __init__(self):
        self.coin_names = UPBIT.get_all_coin_names()

    def record(self, base_datetime, coin_ticker_name):
        time_str_hour = base_datetime.split(" ")[1].split(":")[0]
        time_str_minute = base_datetime.split(" ")[1].split(":")[1]

        if time_str_hour[0] == "0":
            time_str_hour = time_str_hour[1:]

        if time_str_minute[0] == "0":
            time_str_minute = time_str_minute[1:]

        daily_base_timestamp = int(time_str_hour) * 100 + int(time_str_minute)

        order_book = UPBIT.get_orderbook(tickers=coin_ticker_name)[0]

        order_book_units = order_book["orderbook_units"]
        ask_price_lst = []
        ask_size_lst = []
        bid_price_lst = []
        bid_size_lst = []
        for item in order_book_units:
            ask_price_lst.append(item["ask_price"])
            ask_size_lst.append(item["ask_size"])
            bid_price_lst.append(item["bid_price"])
            bid_size_lst.append(item["bid_size"])

        collect_timestamp = order_book['timestamp']
        total_ask_size = order_book['total_ask_size']
        total_bid_size = order_book['total_bid_size']

        return {"base_datetime": base_datetime,
                "daily_base_timestamp": daily_base_timestamp,
                "collect_timestamp": collect_timestamp,
                "ask_price_lst": ask_price_lst,
                "ask_size_lst": ask_size_lst,
                "bid_price_lst": bid_price_lst,
                "bid_size_lst": bid_size_lst,
                "total_ask_size": total_ask_size,
                "total_bid_size": total_bid_size}

    def insert_order_book(self, order_book_info):
        with sqlite3.connect(sqlite3_order_book_db_filename, timeout=10, isolation_level=None,
                             check_same_thread=False) as conn:
            cursor = conn.cursor()

            for coin_name in order_book_info:
                cursor.execute(order_book_insert_sql.format(coin_name), (
                    order_book_info[coin_name]["base_datetime"],
                    int(order_book_info[coin_name]["daily_base_timestamp"]),
                    int(order_book_info[coin_name]["collect_timestamp"]),

                    float(order_book_info[coin_name]["ask_price_lst"][0]),
                    float(order_book_info[coin_name]["ask_size_lst"][0]),

                    float(order_book_info[coin_name]["ask_price_lst"][1]),
                    float(order_book_info[coin_name]["ask_size_lst"][1]),

                    float(order_book_info[coin_name]["ask_price_lst"][2]),
                    float(order_book_info[coin_name]["ask_size_lst"][2]),

                    float(order_book_info[coin_name]["ask_price_lst"][3]),
                    float(order_book_info[coin_name]["ask_size_lst"][3]),

                    float(order_book_info[coin_name]["ask_price_lst"][4]),
                    float(order_book_info[coin_name]["ask_size_lst"][4]),

                    float(order_book_info[coin_name]["ask_price_lst"][5]),
                    float(order_book_info[coin_name]["ask_size_lst"][5]),

                    float(order_book_info[coin_name]["ask_price_lst"][6]),
                    float(order_book_info[coin_name]["ask_size_lst"][6]),

                    float(order_book_info[coin_name]["ask_price_lst"][7]),
                    float(order_book_info[coin_name]["ask_size_lst"][7]),

                    float(order_book_info[coin_name]["ask_price_lst"][8]),
                    float(order_book_info[coin_name]["ask_size_lst"][8]),

                    float(order_book_info[coin_name]["ask_price_lst"][9]),
                    float(order_book_info[coin_name]["ask_size_lst"][9]),

                    float(order_book_info[coin_name]["ask_price_lst"][10]),
                    float(order_book_info[coin_name]["ask_size_lst"][10]),

                    float(order_book_info[coin_name]["ask_price_lst"][11]),
                    float(order_book_info[coin_name]["ask_size_lst"][11]),

                    float(order_book_info[coin_name]["ask_price_lst"][12]),
                    float(order_book_info[coin_name]["ask_size_lst"][12]),

                    float(order_book_info[coin_name]["ask_price_lst"][13]),
                    float(order_book_info[coin_name]["ask_size_lst"][13]),

                    float(order_book_info[coin_name]["ask_price_lst"][14]),
                    float(order_book_info[coin_name]["ask_size_lst"][14]),

                    float(order_book_info[coin_name]["bid_price_lst"][0]),
                    float(order_book_info[coin_name]["bid_size_lst"][0]),

                    float(order_book_info[coin_name]["bid_price_lst"][1]),
                    float(order_book_info[coin_name]["bid_size_lst"][1]),

                    float(order_book_info[coin_name]["bid_price_lst"][2]),
                    float(order_book_info[coin_name]["bid_size_lst"][2]),

                    float(order_book_info[coin_name]["bid_price_lst"][3]),
                    float(order_book_info[coin_name]["bid_size_lst"][3]),

                    float(order_book_info[coin_name]["bid_price_lst"][4]),
                    float(order_book_info[coin_name]["bid_size_lst"][4]),

                    float(order_book_info[coin_name]["bid_price_lst"][5]),
                    float(order_book_info[coin_name]["bid_size_lst"][5]),

                    float(order_book_info[coin_name]["bid_price_lst"][6]),
                    float(order_book_info[coin_name]["bid_size_lst"][6]),

                    float(order_book_info[coin_name]["bid_price_lst"][7]),
                    float(order_book_info[coin_name]["bid_size_lst"][7]),

                    float(order_book_info[coin_name]["bid_price_lst"][8]),
                    float(order_book_info[coin_name]["bid_size_lst"][8]),

                    float(order_book_info[coin_name]["bid_price_lst"][9]),
                    float(order_book_info[coin_name]["bid_size_lst"][9]),

                    float(order_book_info[coin_name]["bid_price_lst"][10]),
                    float(order_book_info[coin_name]["bid_size_lst"][10]),

                    float(order_book_info[coin_name]["bid_price_lst"][11]),
                    float(order_book_info[coin_name]["bid_size_lst"][11]),

                    float(order_book_info[coin_name]["bid_price_lst"][12]),
                    float(order_book_info[coin_name]["bid_size_lst"][12]),

                    float(order_book_info[coin_name]["bid_price_lst"][13]),
                    float(order_book_info[coin_name]["bid_size_lst"][13]),

                    float(order_book_info[coin_name]["bid_price_lst"][14]),
                    float(order_book_info[coin_name]["bid_size_lst"][14]),

                    float(order_book_info[coin_name]["total_ask_size"]),
                    float(order_book_info[coin_name]["total_bid_size"])
                ))
            conn.commit()


if __name__ == "__main__":
    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    base_datetime = current_time_str[:-3] + ":00"

    upbit_order_book_recorder = UpbitOrderBookRecorder()

    current_time = time.time()

    order_book_info = {}
    for coin_name in upbit_order_book_recorder.coin_names:
        order_book_info[coin_name] = upbit_order_book_recorder.record(
            base_datetime=base_datetime,
            coin_ticker_name="KRW-" + coin_name
        )

    upbit_order_book_recorder.insert_order_book(order_book_info)

    elapsed_time = time.time() - current_time
    logger.info("{0} - Elapsed Time: {1} - Num of coins: {2}".format(base_datetime, elapsed_time, len(order_book_info)))
