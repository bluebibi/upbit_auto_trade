import datetime as dt
import sqlite3

from pytz import timezone

from common.global_variables import *


select_all_bought_coin_names_sql = "SELECT * FROM BUY_SELL WHERE status=? or status=?;"
update_trail_coin_info_sql = "UPDATE BUY_SELL SET trail_datetime=?, trail_price=?, trail_rate=?, status=? WHERE " \
                             "coin_ticker_name=? and buy_datetime=?;"


def select_all_bought_coin_names():
    with sqlite3.connect(sqlite3_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(
            select_all_bought_coin_names_sql, (CoinStatus.bought.value, CoinStatus.trailed.value)
        )

        buy_trail_coin_info = {}
        buy_coin_names = []
        rows = cursor.fetchall()
        for row in rows:
            coin_name = row[0]
            buy_datetime = dt.datetime.strptime(row[1], fmt.replace("T", " "))
            cnn_prob = float(row[2])
            lstm_prob = float(row[3])
            buy_price = float(row[4])
            buy_trail_coin_info["KRW-"+coin_name] = {
                "buy_datetime_str": row[1],
                "buy_datetime": buy_datetime,
                "cnn_prob": cnn_prob,
                "lstm_prob": lstm_prob,
                "buy_price": buy_price
            }
            buy_coin_names.append("KRW-"+coin_name)

        conn.commit()

    print(buy_trail_coin_info)

    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    now_datetime = dt.datetime.strptime(now_str, fmt.replace("T", " "))

    if buy_coin_names:
        prices = UPBIT.get_current_price(buy_coin_names)
        trail_coin_info = {}
        for coin_ticker_name in buy_trail_coin_info:
            trail_price = float(prices[coin_ticker_name])
            trail_rate = 100 * trail_price / buy_trail_coin_info[coin_ticker_name]["buy_price"]

            buy_datetime = buy_trail_coin_info["buy_datetime"]
            time_diff = now_datetime - buy_datetime
            time_diff_minutes = time_diff.seconds / 60

            if time_diff_minutes > 180:
                if trail_rate > SELL_RATE:
                    coin_status = CoinStatus.success_sold.value
                elif trail_rate > 0.0:
                    coin_status = CoinStatus.gain_sold.value
                else:
                    coin_status = CoinStatus.loss_sold.value
            else:
                coin_status = CoinStatus.trailed.value

            trail_coin_info[coin_ticker_name] = {
                "trail_datetime": current_time_str,
                "trail_price": trail_price,
                "trail_rate": trail_rate,
                "status": coin_status,
                "buy_datetime": buy_trail_coin_info["buy_datetime_str"]
            }

        update_coin_info(trail_coin_info)


def update_coin_info(trail_coin_info):
    with sqlite3.connect(sqlite3_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        for coin_ticker_name in trail_coin_info:
            cursor.execute(update_trail_coin_info_sql, (
                trail_coin_info[coin_ticker_name]["trail_datetime"],
                trail_coin_info[coin_ticker_name]["trail_price"],
                trail_coin_info[coin_ticker_name]["trail_rate"],
                trail_coin_info[coin_ticker_name]["status"],
                coin_ticker_name,
                trail_coin_info[coin_ticker_name]["buy_datetime"]
            ))
        conn.commit()


def main():
    select_all_bought_coin_names()


if __name__ == "__main__":
    main()
