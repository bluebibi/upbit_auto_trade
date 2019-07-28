import glob
import locale
import sqlite3

import sys, os
sys.path.append(os.getcwd())

from common.global_variables import *
from predict.model_cnn import CNN
from predict.model_rnn import LSTM
from upbit.upbit_data import UpbitData
from pytz import timezone
from common.utils import *
from common.logger import get_logger

logger = get_logger("buy_logger")

if os.getcwd().endswith("upbit_auto_trade"):
    pass
elif os.getcwd().endswith("predict"):
    os.chdir("..")
else:
    pass

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

select_coin_ticker_name_by_status_sql = "SELECT coin_ticker_name FROM BUY_SELL WHERE status=0 or status=1;"
select_by_datetime = "SELECT * FROM {0} WHERE datetime='{1}';"
insert_buy_try_coin_info = "INSERT INTO BUY_SELL (coin_ticker_name, buy_datetime, cnn_prob, lstm_prob, buy_price, status) VALUES (?, ?, ?, ?, ?, ?);"


def get_coin_ticker_name_by_status():
    coin_ticker_name_list = []
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        rows = cursor.execute(select_coin_ticker_name_by_status_sql)

        for coin_ticker_name in rows:
            coin_ticker_name_list.append(coin_ticker_name[0].replace("KRW-", ""))
        conn.commit()
    return coin_ticker_name_list


def get_good_quality_coin_names_for_buy():
    cnn_models = {}
    cnn_files = glob.glob('./models/CNN/*.pt')

    lstm_models = {}
    lstm_files = glob.glob('./models/LSTM/*.pt')

    for f in cnn_files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("./models/CNN/", "")
            model = CNN(input_width=INPUT_SIZE, input_height=WINDOW_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f, map_location=DEVICE))
            model.eval()
            cnn_models[coin_name] = model

    for f in lstm_files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("./models/LSTM/", "")
            model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f, map_location=DEVICE))
            model.eval()
            lstm_models[coin_name] = model

    return cnn_models, lstm_models


def get_db_right_time_coin_names():
    coin_names = {}
    now = dt.datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    current_time_str = current_time_str[:-4] + "0:00"

    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()
        all_coin_names = UPBIT.get_all_coin_names()
        for coin_name in all_coin_names:
            cursor.execute(select_by_datetime.format("KRW_" + coin_name, current_time_str))
            row = cursor.fetchall()
            if len(row) == 1:
                coin_names[coin_name] = current_time_str
        conn.commit()

    return coin_names


def evaluate_coin_by_models(model, coin_name, model_type):
    upbit_data = UpbitData(coin_name)
    x = upbit_data.get_buy_for_data(model_type=model_type)

    out = model.forward(x)
    out = torch.sigmoid(out)
    t = torch.Tensor([0.5]).to(DEVICE)
    output_index = (out > t).float() * 1

    prob = out.item()
    idx = int(output_index.item())

    if idx and prob > 0.9:
        return prob
    else:
        return None


def insert_buy_coin_info(buy_try_coin_info):
    msg_str = "*** BUY\n"
    with sqlite3.connect(sqlite3_price_info_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        for coin_ticker_name in buy_try_coin_info:
            cursor.execute(insert_buy_try_coin_info, (
                coin_ticker_name,
                buy_try_coin_info[coin_ticker_name]["right_time"],
                convert_unit_2(float(buy_try_coin_info[coin_ticker_name]["cnn_prob"])),
                convert_unit_2(float(buy_try_coin_info[coin_ticker_name]["lstm_prob"])),
                float(buy_try_coin_info[coin_ticker_name]["buy_price"]),
                CoinStatus.bought.value
            ))
            msg_str += "[{0}, {1}, {2}, {3}, {4}, {5}]\n".format(
                coin_ticker_name,
                buy_try_coin_info[coin_ticker_name]["right_time"],
                convert_unit_2(float(buy_try_coin_info[coin_ticker_name]["cnn_prob"])),
                convert_unit_2(float(buy_try_coin_info[coin_ticker_name]["lstm_prob"])),
                locale.format_string("%.2f", float(buy_try_coin_info[coin_ticker_name]["buy_price"]), grouping=True),
                coin_status_to_hangul(CoinStatus.bought.value)
            )
        conn.commit()
    return msg_str


def main():
    good_cnn_models, good_lstm_models = get_good_quality_coin_names_for_buy()

    right_time_coin_names = get_db_right_time_coin_names()

    already_coin_ticker_names = get_coin_ticker_name_by_status()

    target_coin_names = (set(good_cnn_models) & set(good_lstm_models) & set(right_time_coin_names)) - set(already_coin_ticker_names)

    logger.info("*** CNN: {0}, LSTM: {1}, Right Time Coins: {2}, Target Coins: {3} ***".format(
        len(good_cnn_models),
        len(good_lstm_models),
        len(right_time_coin_names),
        target_coin_names
    ))

    if len(target_coin_names) > 0:
        buy_try_coin_info = {}
        buy_try_coin_names = []

        for coin_name in target_coin_names:
            cnn_prob = evaluate_coin_by_models(
                model=good_cnn_models[coin_name],
                coin_name=coin_name,
                model_type="CNN"
            )

            lstm_prob = evaluate_coin_by_models(
                model=good_lstm_models[coin_name],
                coin_name=coin_name,
                model_type="LSTM"
            )

            logger.info("{0} --> CNN Probability:{1:.4f}, LSTM Probability:{2:.4f}".format(coin_name, cnn_prob, lstm_prob))

            if cnn_prob and lstm_prob:
                # coin_name --> right_time, prob
                buy_try_coin_info["KRW-" + coin_name] = {
                    "right_time": right_time_coin_names[coin_name],
                    "cnn_prob": cnn_prob,
                    "lstm_prob": lstm_prob
                }
                buy_try_coin_names.append("KRW-" + coin_name)

        if buy_try_coin_names:
            prices = UPBIT.get_current_price(buy_try_coin_names)

            for coin_ticker_name in buy_try_coin_info:
                buy_try_coin_info[coin_ticker_name]["buy_price"] = prices[coin_ticker_name]

        if len(buy_try_coin_info) > 0:
            msg_str = insert_buy_coin_info(buy_try_coin_info)
            msg_str += " @ " + SOURCE
            SLACK.send_message("me", msg_str)

            logger.info("{0}".format(msg_str))


if __name__ == "__main__":
    main()
