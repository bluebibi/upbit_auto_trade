import sqlite3

from pytz import timezone

from common.global_variables import *
from common.utils import *
from common.logger import get_logger

logger = get_logger("sell_logger")

if os.getcwd().endswith("predict"):
    os.chdir("..")

select_all_bought_coin_names_sql = "SELECT * FROM BUY_SELL WHERE status=? or status=?;"
update_trail_coin_info_sql = "UPDATE BUY_SELL SET trail_datetime=?, trail_price=?, trail_rate=?, status=? WHERE " \
                             "coin_ticker_name=? and buy_datetime=?;"


class Seller:
    def select_all_bought_coin_names(self):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute(
                select_all_bought_coin_names_sql, (CoinStatus.bought.value, CoinStatus.trailed.value)
            )

            buy_trail_coin_info = {}
            buy_trail_coin_names = []
            rows = cursor.fetchall()
            conn.commit()

        for row in rows:
            coin_ticker_name = row[1]
            buy_datetime = dt.datetime.strptime(row[2], fmt.replace("T", " "))
            cnn_prob = float(row[3])
            lstm_prob = float(row[4])
            buy_price = float(row[5])
            buy_trail_coin_info[coin_ticker_name] = {
                "buy_datetime_str": row[2],
                "buy_datetime": buy_datetime,
                "cnn_prob": cnn_prob,
                "lstm_prob": lstm_prob,
                "buy_price": buy_price
            }
            buy_trail_coin_names.append(coin_ticker_name)

        now = dt.datetime.now(timezone('Asia/Seoul'))
        now_str = now.strftime(fmt)
        current_time_str = now_str.replace("T", " ")
        now_datetime = dt.datetime.strptime(now_str, fmt)

        msg_str = ""
        if buy_trail_coin_names:
            prices = UPBIT.get_current_price(buy_trail_coin_names)
            trail_coin_info = {}
            for coin_ticker_name in buy_trail_coin_info:
                trail_price = float(prices[coin_ticker_name])
                buy_price = buy_trail_coin_info[coin_ticker_name]["buy_price"]
                trail_rate = (trail_price - buy_price) / buy_price

                buy_datetime = buy_trail_coin_info[coin_ticker_name]["buy_datetime"]
                time_diff = now_datetime - buy_datetime
                time_diff_minutes = time_diff.seconds / 60

                if trail_rate > SELL_RATE:
                    coin_status = CoinStatus.success_sold.value
                else:
                    if time_diff_minutes > FUTURE_TARGET_SIZE * 10:
                        if trail_rate > TRANSACTION_FEE_RATE:
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
                    "buy_datetime": buy_trail_coin_info[coin_ticker_name]["buy_datetime_str"]
                }

                if coin_status == CoinStatus.success_sold.value or coin_status == CoinStatus.gain_sold.value:
                    msg_str += "[{0}, {1}, {2}%, {3}]\n".format(
                        coin_ticker_name,
                        trail_price,
                        convert_unit_2(trail_rate * 100),
                        coin_status_to_hangul(coin_status)
                    )

            self.update_coin_info(trail_coin_info)
        return msg_str

    def update_coin_info(self, trail_coin_info):
        with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
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

    def try_to_sell(self):
        msg_str = self.select_all_bought_coin_names()

        if msg_str:
            msg_str = "### SELL\n" + msg_str + " @ " + SOURCE

            pusher = PushSlack(SLACK_WEBHOOK_URL_1, SLACK_WEBHOOK_URL_2)
            pusher.send_message("me", msg_str)

            logger.info("{0}".format(msg_str))


if __name__ == "__main__":
    seller = Seller()
    while True:
        seller.try_to_sell()
        time.sleep(SELL_PERIOD)
